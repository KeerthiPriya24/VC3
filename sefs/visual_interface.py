"""Visual Interface – Streamlit-based 2D semantic visualization with V2 features."""

from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from sefs.config import SEFSConfig
from sefs.database import Database
from sefs.models import FileMetadata, ClusterInfo, ClusterUpdate
from sefs.os_synchronizer import OSSynchronizer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEFS – Semantic Entropy File System",
    page_icon="🗂️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Try auto-refresh (optional dependency)
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="sefs_refresh")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Curated cluster colour palette
_PALETTE = [
    "#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6",
    "#8b5cf6", "#ec4899", "#14b8a6", "#ef4444", "#06b6d4",
    "#a855f7", "#22c55e", "#e11d48", "#0ea5e9", "#84cc16",
]


def _get_db_and_config(config_path: str) -> tuple[Database, SEFSConfig]:
    cfg = SEFSConfig.load(Path(config_path))
    return Database(cfg.database_path), cfg


def _project_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2-D via t-SNE (or PCA for tiny sets)."""
    n = embeddings.shape[0]
    if n < 2:
        return np.zeros((n, 2))
    perplexity = min(30, max(1, n - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    return tsne.fit_transform(embeddings)


def _open_file(path: str) -> None:
    path = str(Path(path).resolve())
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            subprocess.Popen(["start", "", path], shell=True)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        st.error(f"Failed to open file: {e}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(
        """
        <style>
        .main { background: #0f172a; }
        h1, h2, h3 { color: #e2e8f0; }
        .stSidebar { background: #1e293b; }
        .stSidebar .stMarkdown { color: #94a3b8; }
        div[data-testid="metric-container"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🗂️ SEFS — Semantic Entropy File System")

    # Load config and DB early
    # Try CWD first, then default
    potential_paths = [
        Path.cwd() / ".sefs" / "config.json",
        Path.home() / "sefs_root" / ".sefs" / "config.json",
    ]
    
    config_path_default = str(potential_paths[1])
    for p in potential_paths:
        if p.exists():
            config_path_default = str(p)
            break
    
    # Allow override via query param or default
    if "config_path" not in st.session_state:
        st.session_state.config_path = config_path_default

    try:
        db, config = _get_db_and_config(st.session_state.config_path)
    except Exception as exc:
        # Fallback input
        with st.sidebar:
            st.error(f"Cannot load default config.")
            alt_path = st.text_input("Path to .sefs/config.json", value=config_path_default)
            if st.button("Load"):
                st.session_state.config_path = alt_path
                st.rerun()
        st.info("Run `python -m sefs.main <root_dir>` first to initialize.")
        return

    all_files = db.get_all_files()
    clusters = db.get_all_clusters()

    # ---------------------------------------------------------------------------
    # Sidebar Controls
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("SEFS Control")
        
        # --- Re-Run Section (Prominent) ---
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            if st.button("🔄 Re-Scan Now", type="primary", use_container_width=True):
                db.set_config("trigger_scan", "true")
                st.toast("✅ Re-scan triggered! Backend will process shortly.")
                time.sleep(1)
                st.rerun()
        with col_run2:
            if st.button("🧹 Full Re-Analyze", use_container_width=True, help="Clear all embeddings and re-analyze from scratch"):
                db._conn.execute("UPDATE files SET embedding = NULL")
                db._conn.commit()
                db.set_config("trigger_scan", "true")
                st.toast("🔄 Full re-analysis triggered! This may take a moment.")
                time.sleep(1)
                st.rerun()
        
        # --- File Upload Section with Approval Dialog ---
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("Drop a file to organize", type=['txt', 'pdf', 'md', 'log', 'rst'])
        if uploaded_file is not None:
            upload_key = f"up_{uploaded_file.name}_{uploaded_file.size}"
            # Only show dialog if this file hasn't been processed yet
            if upload_key not in st.session_state:
                save_path = Path(config.root_directory) / uploaded_file.name
                in_db = any(f.filename == uploaded_file.name for f in all_files)

                if in_db:
                    st.warning("File already tracked in SEFS.")
                elif save_path.exists():
                    st.info("File already at root, waiting for sync.")
                else:
                    # Show approval dialog
                    st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
                    st.markdown("Would you like SEFS to **read and classify** this file?")
                    col_approve, col_deny = st.columns(2)
                    with col_approve:
                        if st.button("✅ Approve", key=f"approve_{upload_key}", type="primary", use_container_width=True):
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.session_state[upload_key] = "approved"
                            # Trigger backend scan so file gets classified
                            db.set_config("trigger_scan", "true")
                            st.toast(f"✅ {uploaded_file.name} saved & classification started!")
                            # Set a flag so the page auto-refreshes while clustering runs
                            st.session_state["awaiting_cluster"] = time.time()
                            time.sleep(1)
                            st.rerun()
                    with col_deny:
                        if st.button("❌ Deny", key=f"deny_{upload_key}", use_container_width=True):
                            st.session_state[upload_key] = "denied"
                            st.toast("File upload denied. Nothing was saved.")
                            time.sleep(0.5)
                            st.rerun()
            elif st.session_state.get(upload_key) == "denied":
                st.caption("⛔ This file was denied. Upload a different file to proceed.")

        # Auto-reload while awaiting clustering after upload
        if st.session_state.get("awaiting_cluster"):
            elapsed = time.time() - st.session_state["awaiting_cluster"]
            if elapsed < 30:  # Poll for up to 30 seconds
                time.sleep(2)
                st.rerun()
            else:
                # Stop polling after 30s
                del st.session_state["awaiting_cluster"]
        
        st.markdown("---")

        # Search with content matching
        search_query = st.text_input("🔍 Search files", placeholder="Search by filename or content...")
        
        if search_query:
            st.caption(f"Searching for: *{search_query}*")

        # No folder filter — always show all
        selected_folder = "All"

        st.caption(f"Active Folders: {len(clusters)} | Total Files: {len(all_files)}")

        # --- Manual Approval Mode ---
        st.markdown("---")
        manual_mode = db.get_config("manual_mode") == "true"
        new_mode = st.toggle("Manual Approval Mode", value=manual_mode, help="If enabled, files won't be moved automatically.")
        if new_mode != manual_mode:
            db.set_config("manual_mode", "true" if new_mode else "false")
            st.rerun()

        if manual_mode:
            st.markdown("---")
            st.subheader("⚠️ Pending Actions")
            
            # Find files that need moving based on PROPOSED_CLUSTER_ID
            pending_moves = []
            for f in all_files:
                target_cid = f.proposed_cluster_id
                if target_cid:
                    cluster = next((c for c in clusters if c.cluster_id == target_cid), None)
                    if cluster:
                        target_name = cluster.folder_name
                        expected_parent = Path(config.root_directory) / target_name
                        if f.file_path.parent.resolve() != expected_parent.resolve():
                            # Check if this specific target was already denied
                            if f.denied_folder_path == target_name:
                                continue
                            pending_moves.append((f, target_name))
                else:
                    # No proposed cluster -> should be at root
                    if f.file_path.parent.resolve() != Path(config.root_directory).resolve():
                        pending_moves.append((f, ".")) # "." means root
            
            if pending_moves:
                st.warning(f"{len(pending_moves)} moves pending.")
                
                # Single sync helper
                def apply_sync(moves_dict):
                    sync = OSSynchronizer(Path(config.root_directory), db)
                    for fp, target_path in moves_dict.items():
                        f_meta = next((f for f in all_files if str(f.file_path) == str(fp)), None)
                        if f_meta and f_meta.proposed_cluster_id:
                            db.update_file_cluster(fp, f_meta.proposed_cluster_id, 1.0)
                            db.update_proposed_cluster(fp, None)
                        elif str(target_path) == ".":
                             db.update_file_cluster(fp, None, 0.0)
                             db.update_proposed_cluster(fp, None)

                    update = ClusterUpdate(new_clusters={c.cluster_id: c for c in clusters}, file_moves=moves_dict)
                    try:
                        sync.apply_cluster_update(update)
                        # Trigger backend re-scan to reconcile state
                        db.set_config("trigger_scan", "true")
                        st.toast("✅ Moves applied successfully!")
                        time.sleep(0.5)
                    except Exception as e:
                        st.error(f"Move failed: {e}")
                    st.rerun()

                if st.button("Apply All Moves", type="primary"):
                    apply_sync({f.file_path: Path(target) for f, target in pending_moves})

                st.write("Individual Moves:")
                for i, (f, target) in enumerate(pending_moves):
                    with st.expander(f"{f.filename} → {target}"):
                        st.caption(f"From: `{f.file_path.parent.name or 'root'}`")
                        c1, c2 = st.columns(2)
                        if c1.button("Approve", key=f"app_{i}"):
                            apply_sync({f.file_path: Path(target)})
                        
                        if c2.button("Deny", key=f"deny_{i}"):
                            db.deny_file_cluster(f.file_path, target)
                            st.toast("Move denied permanently.")
                            st.rerun()
            else:
                st.info("Everything in sync.")

    # ---------------------------------------------------------------------------
    # Global View
    # ---------------------------------------------------------------------------
    st.subheader("Global View")

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------

    # Apply filters
    filtered_files = list(all_files)

    if search_query:
        q = search_query.lower().strip()
        search_results = []
        for f in filtered_files:
            # Match against filename
            if q in f.filename.lower():
                search_results.append(f)
                continue
            # Match against content summary
            if f.content_summary and q in f.content_summary.lower():
                search_results.append(f)
                continue
            # Match against file type
            if q in f.file_type.lower():
                search_results.append(f)
                continue
        filtered_files = search_results

    # Build a fast lookup set for filtered file paths
    filtered_paths_set = {str(f.file_path) for f in filtered_files}

    st.markdown(f"**Showing {len(filtered_files)} of {len(all_files)} file(s)**")

    if not filtered_files:
        st.info("No files match your criteria.")
        # Only stop drawing graph, but proceed to table if needed
    else:
        # Prepare data for Plotly
        # We need to project ALL embeddings to keep the map stable, then highlight filtered
        valid_files = [f for f in all_files if f.embedding is not None]
        if len(valid_files) < 2:
            if all_files:
                st.info("🔄 **Re-analysis in progress...** The semantic map will appear once your files have been processed with the new model. Please ensure the backend script is running.")
            else:
                st.warning("Not enough files to visualize (need at least 2).")
        else:
            all_embeddings = np.array([f.embedding for f in valid_files])
            projections = _project_2d(all_embeddings)

            # Create lookup
            cid_to_folder = {c.cluster_id: c.folder_name for c in clusters}

            # Files
            plot_data = []
            for idx, f in enumerate(valid_files):
                folder = cid_to_folder.get(f.cluster_id, "Uncategorized")
                is_visible = str(f.file_path) in filtered_paths_set
                opacity = 1.0 if is_visible else 0.15
                
                plot_data.append({
                    "x": projections[idx, 0],
                    "y": projections[idx, 1],
                    "label": f.filename,
                    "folder": folder,
                    "size": 10,
                    "type": "File",
                    "path": str(f.file_path),
                    "opacity": opacity
                })

            df_plot = pd.DataFrame(plot_data)

            # Folders (Centroids for visual reference)
            folder_data = []
            for folder in df_plot['folder'].unique():
                subset = df_plot[df_plot['folder'] == folder]
                folder_data.append({
                    "x": subset['x'].mean(),
                    "y": subset['y'].mean(),
                    "label": f"📁 {folder}",
                    "folder": folder,
                    "size": 30,
                    "type": "Folder",
                    "path": "",
                    "opacity": 0.8
                })
            
            df_folders = pd.DataFrame(folder_data)
            df_combined = pd.concat([df_plot, df_folders])

            # Assign consistent colors
            unique_folders = sorted(df_combined['folder'].unique())
            color_map = {folder: _PALETTE[i % len(_PALETTE)] for i, folder in enumerate(unique_folders)}

            fig = px.scatter(
                df_combined,
                x="x",
                y="y",
                color="folder",
                size="size",
                size_max=25,
                text=df_combined.apply(lambda r: r['label'] if r['type'] == 'Folder' else '', axis=1),
                hover_data={"x": False, "y": False, "label": True, "type": True, "folder": True, "size": False},
                custom_data=["path", "type", "label"],
                title="Semantic Map (Click file to open)",
                template="plotly_dark",
                color_discrete_map=color_map,
                height=700
            )

            fig.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1, color='rgba(255, 255, 255, 0.4)')),
                selector=dict(type='scatter')
            )

            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=True,
                legend_title_text='Folders'
            )

            # Display chart (no on_select to prevent file-opening loops)
            st.plotly_chart(fig, use_container_width=True)

    # Detect model change for re-analysis notice
    current_model = db.get_config("current_model")
    if current_model != config.embedding_model:
        st.warning(f"Embedding model upgraded to `{config.embedding_model}`. A full re-analysis is required for maximum accuracy.")
        if st.button("Run Full Re-Analysis Now"):
            db.set_config("current_model", config.embedding_model)
            # Sync to config.json so backend picks up the new model name
            config.save() 
            # Clear embeddings to force re-analysis
            db._conn.execute("UPDATE files SET embedding = NULL")
            db._conn.commit()
            st.info("Embeddings cleared and config updated. Backend will re-analyze files on next scan.")
            st.rerun()

    # ---------------------------------------------------------------------------
    # Data fetch
    # ---------------------------------------------------------------------------
    st.subheader("File Library")
    
    # Create lookup for mapping files to clusters
    cid_to_folder = {c.cluster_id: c.folder_name for c in clusters}
    
    # Group files by semantic folder (cluster), not physical location
    grouped_files = {}
    for f in filtered_files:
        folder = cid_to_folder.get(f.cluster_id, "[Uncategorized]")
        if folder not in grouped_files:
            grouped_files[folder] = []
        grouped_files[folder].append(f)
        
    # Sort folders alphabetically
    sorted_folders = sorted(grouped_files.keys())
    
    if not sorted_folders:
        st.info("No files to display.")
    
    for folder in sorted_folders:
        files_in_folder = grouped_files[folder]
        with st.expander(f"📁 {folder} ({len(files_in_folder)} file(s))", expanded=True):
            for fi, f in enumerate(files_in_folder):
                col_name, col_size, col_date, col_open = st.columns([3, 1, 2, 1])
                col_name.markdown(f"**{f.filename}**")
                col_size.caption(f"{f.file_size / 1024:.1f} KB")
                col_date.caption(f.modification_date.strftime("%Y-%m-%d %H:%M"))
                if col_open.button("📂 Open", key=f"open_{folder}_{fi}_{f.filename}"):
                    _open_file(str(f.file_path))

    # ---------------------------------------------------------------------------
    # File Management (New)
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Manage Files")
    
    if filtered_files:
        # Create a mapping for selection
        file_options = {}
        for f in filtered_files:
            # Find folder again (could optimize this, but okay for UI)
            folder = cid_to_folder.get(f.cluster_id, "Uncategorized")
            label = f"{f.filename} ({folder})"
            file_options[label] = f
            
        selected_label = st.selectbox("Select File to Manage", list(file_options.keys()))
        if selected_label:
            target_f = file_options[selected_label]
            target_path = Path(target_f.file_path)
            
            c1, c2 = st.columns(2)
            with c1:
                new_name = st.text_input("New Name", value=target_f.filename)
                if st.button("Rename File", key="rename_confirm"):
                    current_p = Path(target_f.file_path)
                    if not current_p.exists():
                        st.error("File no longer exists at expected path.")
                    elif new_name == target_f.filename:
                        st.info("Name unchanged.")
                    else:
                        new_path = current_p.parent / new_name
                        if new_path.exists():
                            st.error("A file with that name already exists!")
                        else:
                            try:
                                current_p.rename(new_path)
                                # Update BOTH file_path and filename in the DB
                                db.rename_file_path(current_p, new_path)
                                db._conn.execute(
                                    "UPDATE files SET filename = ? WHERE file_path = ?",
                                    (new_name, str(new_path)),
                                )
                                db._conn.commit()
                                # Trigger re-scan so backend reconciles
                                db.set_config("trigger_scan", "true")
                                st.success(f"Renamed to {new_name}")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Rename failed: {e}")

            with c2:
                st.write("") # Spacer
                st.write("")
                if st.button("Delete File", type="primary", key="delete_confirm"):
                    current_p = Path(target_f.file_path)
                    if not current_p.exists():
                        st.error("File already gone or path stale.")
                    else:
                        try:
                            current_p.unlink()
                            db.delete_file(current_p)
                            st.warning(f"Deleted {target_f.filename}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
    else:
        st.info("No files available to manage.")


if __name__ == "__main__":
    main()
