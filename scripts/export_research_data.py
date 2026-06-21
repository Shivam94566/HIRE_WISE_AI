import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine

# Set professional research styles using standard matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
EXPORT_DIR = BASE_DIR / "exports"
GRAPH_DIR = EXPORT_DIR / "graphs"
EXPORT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

def get_db_connection():
    """Returns a SQLAlchemy engine connecting to PostgreSQL (from env) or local SQLite."""
    # Check .env first
    env_path = BASE_DIR / ".env"
    db_uri = None
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("DATABASE_URL="):
                    db_uri = line.split("DATABASE_URL=")[1].strip()
                    break
    
    if not db_uri:
        db_uri = os.environ.get("DATABASE_URL")

    if db_uri:
        # Normalize postgres protocol for SQLAlchemy
        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        print(f"[INFO] Connecting to Cloud Database: {db_uri.split('@')[-1]}")
        return create_engine(db_uri)
    else:
        db_path = BASE_DIR / "database" / "hirewise.db"
        print(f"[INFO] Connecting to Local SQLite Database: {db_path}")
        return create_engine(f"sqlite:///{db_path.as_posix()}")

def load_data(engine):
    """Loads all relevant tables from the database into Pandas DataFrames."""
    tables = [
        "users", "resume_uploads", "interview_sessions", 
        "responses", "questions", "session_skill_states", 
        "session_skill_history", "performance_reports", 
        "chat_sessions", "chat_messages", "practice_history"
    ]
    data = {}
    for table in tables:
        try:
            data[table] = pd.read_sql_table(table, engine)
            print(f"[SUCCESS] Loaded table '{table}' with {len(data[table])} rows.")
        except Exception as e:
            print(f"[WARNING] Could not load table '{table}': {e}")
            data[table] = pd.DataFrame()
    return data

def flatten_json_telemetry(sessions_df):
    """Flattens the JSON decision_log and competency_map from interview_sessions."""
    decision_records = []
    competency_records = []

    for _, row in sessions_df.iterrows():
        sess_id = row['id']
        u_id = row['user_id']
        mode = row.get('experiment_mode', 'fixed')

        # 1. Parse decision log (Fisher Information and selection reasons per turn)
        dec_log = row.get('decision_log')
        if dec_log and isinstance(dec_log, str) and dec_log.strip():
            try:
                log_data = json.loads(dec_log)
                for entry in log_data:
                    entry['session_id'] = sess_id
                    entry['user_id'] = u_id
                    entry['experiment_mode'] = mode
                    decision_records.append(entry)
            except Exception as e:
                print(f"[WARNING] Error parsing decision log for session {sess_id}: {e}")

        # 2. Parse competency map (skill bounds and uncertainties)
        comp_map = row.get('competency_map')
        if comp_map and isinstance(comp_map, str) and comp_map.strip():
            try:
                map_data = json.loads(comp_map)
                for skill_name, meta in map_data.items():
                    competency_records.append({
                        "session_id": sess_id,
                        "user_id": u_id,
                        "experiment_mode": mode,
                        "skill_name": skill_name,
                        "estimated_score": meta.get("score"),
                        "uncertainty_sigma": meta.get("uncertainty"),
                        "confidence_level": meta.get("confidence_level"),
                        "inferred_boundary": meta.get("boundary")
                    })
            except Exception as e:
                print(f"[WARNING] Error parsing competency map for session {sess_id}: {e}")

    dec_df = pd.DataFrame(decision_records)
    comp_df = pd.DataFrame(competency_records)
    
    return dec_df, comp_df

def run_statistical_analysis(data, dec_df, comp_df):
    """Computes statistical metrics (correlations, learning curves, group comparisons) for research."""
    stats = {}
    
    # Analysis 1: Communication vs Technical Score Correlation
    sessions = data["interview_sessions"]
    if not sessions.empty and "communication_score" in sessions.columns and "technical_score" in sessions.columns:
        valid_sess = sessions[sessions['status'] == 'completed']
        if len(valid_sess) > 1:
            corr = valid_sess['communication_score'].corr(valid_sess['technical_score'])
            stats["comm_tech_correlation"] = corr
            print(f"[STAT] Pearson Correlation (Comm vs Tech): {corr:.3f}")

    # Analysis 2: Performance metrics across Experiment Modes
    if not sessions.empty and "experiment_mode" in sessions.columns and "overall_score" in sessions.columns:
        valid_sess = sessions[sessions['status'] == 'completed']
        group_means = valid_sess.groupby('experiment_mode')['overall_score'].agg(['count', 'mean', 'std']).reset_index()
        stats["experiment_modes_summary"] = group_means
        print("\n[STAT] Overall Score Summary by Experiment Mode:")
        print(group_means.to_string())

    # Analysis 3: Fisher Information and Uncertainty over Interview Turns
    if not dec_df.empty and "question_index" in dec_df.columns and "uncertainty_sigma" in dec_df.columns:
        turn_summary = dec_df.groupby(['experiment_mode', 'question_index']).agg({
            'uncertainty_sigma': 'mean',
            'fisher_information': 'mean'
        }).reset_index()
        stats["telemetry_by_turn"] = turn_summary
        print("\n[STAT] Turn-by-Turn Telemetry Averaged:")
        print(turn_summary.to_string())

    return stats

def generate_visualizations(data, dec_df, comp_df):
    """Generates research-ready visual figures for the paper and saves them as PNGs."""
    print("\n[INFO] Generating research visualizations...")
    
    # Plot 1: Bayesian Uncertainty Convergence Rate over Turns
    if not dec_df.empty and "question_index" in dec_df.columns and "uncertainty_sigma" in dec_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, group in dec_df.groupby("experiment_mode"):
            # Group by turn and calculate mean
            gp = group.groupby("question_index")["uncertainty_sigma"].mean().reset_index()
            ax.plot(gp["question_index"], gp["uncertainty_sigma"], marker="o", label=name, linewidth=2)
        ax.set_title("Bayesian Competency Estimation: Uncertainty ($\\sigma$) Convergence Rate")
        ax.set_xlabel("Interview Turn (Question Index)")
        ax.set_ylabel("Estimation Uncertainty Standard Error ($\\sigma$)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "competency_uncertainty_convergence.png", dpi=300)
        plt.close()
        print("[VISUAL] Saved: competency_uncertainty_convergence.png")

    # Plot 2: Fisher Information by Turn
    if not dec_df.empty and "question_index" in dec_df.columns and "fisher_information" in dec_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, group in dec_df.groupby("experiment_mode"):
            gp = group.groupby("question_index")["fisher_information"].mean().reset_index()
            ax.plot(gp["question_index"], gp["fisher_information"], marker="s", label=name, linewidth=2)
        ax.set_title("Fisher Information Optimization per Turn")
        ax.set_xlabel("Interview Turn")
        ax.set_ylabel("Fisher Information value ($I(\\theta)$)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "fisher_information_by_turn.png", dpi=300)
        plt.close()
        print("[VISUAL] Saved: fisher_information_by_turn.png")

    # Plot 3: Boxplot comparing Overall Scores between Adaptive and Control Groups
    sessions = data["interview_sessions"]
    if not sessions.empty and "experiment_mode" in sessions.columns and "overall_score" in sessions.columns:
        completed_sessions = sessions[sessions['status'] == 'completed']
        if len(completed_sessions) > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            groups = completed_sessions.groupby("experiment_mode")
            box_data = [group["overall_score"].values for name, group in groups]
            labels = [name for name, group in groups]
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.4)
            for patch in bp['boxes']:
                patch.set_facecolor('#d9f0a3')
                patch.set_alpha(0.8)
                
            # Plot individual data points as scatter for scientific visualization
            for i, (name, group) in enumerate(groups):
                y = group["overall_score"].values
                x = np.random.normal(i + 1, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.6, color="black", edgecolor="none", s=30)
                
            ax.set_title("Candidate Overall Performance Across Selection Policies")
            ax.set_xlabel("Orchestration Mode")
            ax.set_ylabel("Overall Interview Rating (%)")
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(GRAPH_DIR / "overall_score_comparison.png", dpi=300)
            plt.close()
            print("[VISUAL] Saved: overall_score_comparison.png")

    # Plot 4: Correlation Matrix Heatmap
    if not sessions.empty:
        score_cols = [c for c in ["communication_score", "technical_score", "confidence_score", "answer_quality_score", "professionalism_score", "overall_score"] if c in sessions.columns]
        completed_sessions = sessions[sessions['status'] == 'completed']
        if len(completed_sessions) > 1 and len(score_cols) > 1:
            fig, ax = plt.subplots(figsize=(7, 6))
            corr_mat = completed_sessions[score_cols].corr()
            im = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
            
            # Add annotations
            for i in range(len(score_cols)):
                for j in range(len(score_cols)):
                    ax.text(j, i, f"{corr_mat.values[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr_mat.values[i, j]) < 0.7 else "white")
            
            ax.set_xticks(np.arange(len(score_cols)))
            ax.set_yticks(np.arange(len(score_cols)))
            ax.set_xticklabels([c.replace("_", "\n") for c in score_cols])
            ax.set_yticklabels([c.replace("_", " ") for c in score_cols])
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Correlation Matrix of Extracted Interview Ratings")
            plt.tight_layout()
            plt.savefig(GRAPH_DIR / "interview_dimensions_correlation.png", dpi=300)
            plt.close()
            print("[VISUAL] Saved: interview_dimensions_correlation.png")

def save_to_excel(data, dec_df, comp_df, stats):
    """Exports all datasets and flattened tables into a single formatted Excel workbook."""
    excel_path = EXPORT_DIR / "hirewise_research_data.xlsx"
    print(f"\n[INFO] Exporting sheets to Excel: {excel_path}")
    
    # We require openpyxl
    try:
        import openpyxl
    except ImportError:
        print("[WARNING] openpyxl not installed. Attempting installation...")
        import subprocess
        subprocess.run(["pip", "install", "openpyxl"], stdout=subprocess.DEVNULL)
        
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Base database tables
        for tab_name, df in data.items():
            if not df.empty:
                # Truncate large texts for readability
                df_clean = df.copy()
                for col in df_clean.select_dtypes(include=['object']).columns:
                    df_clean[col] = df_clean[col].astype(str).str.slice(0, 1000)
                df_clean.to_excel(writer, sheet_name=tab_name[:30], index=False)
        
        # Derived flattened research telemetry sheets
        if not dec_df.empty:
            dec_df.to_excel(writer, sheet_name="Turn_Decision_Log", index=False)
        if not comp_df.empty:
            comp_df.to_excel(writer, sheet_name="Candidate_Competencies", index=False)
            
        # Analysis stats summary
        if "experiment_modes_summary" in stats:
            stats["experiment_modes_summary"].to_excel(writer, sheet_name="Experiment_Mode_Stats", index=False)
        if "telemetry_by_turn" in stats:
            stats["telemetry_by_turn"].to_excel(writer, sheet_name="Turn_Telemetry_Stats", index=False)
            
    print(f"[SUCCESS] Research dataset exported successfully!")

def main():
    print("=" * 60)
    print("      HIREWISE AI RESEARCH DATA EXPORT & ANALYTICS TOOL      ")
    print("=" * 60)
    
    try:
        engine = get_db_connection()
        data = load_data(engine)
        
        # Flatten complex JSON telemetry columns
        sessions_df = data.get("interview_sessions", pd.DataFrame())
        if not sessions_df.empty:
            dec_df, comp_df = flatten_json_telemetry(sessions_df)
        else:
            dec_df, comp_df = pd.DataFrame(), pd.DataFrame()
            
        # Run analysis and plot graphs
        stats = run_statistical_analysis(data, dec_df, comp_df)
        generate_visualizations(data, dec_df, comp_df)
        
        # Save structured workbook
        save_to_excel(data, dec_df, comp_df, stats)
        print("=" * 60)
        print(f"Audit completed. Data files stored in: {EXPORT_DIR.resolve()}")
        print("=" * 60)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
