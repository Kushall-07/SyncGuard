"""SyncGuard Demo UI (Phase 13B).

Gradio-based demonstration interface for dual-mode deepfake detection:
- Audio-only: Synthetic/spoofed speech detection
- Audio-visual: Temporal synchronization analysis

IMPORTANT LIMITATIONS:
- The AV synchronization model evaluates temporal correspondence between audio and visual streams.
- It is NOT a universal deepfake classifier. Manipulated content can remain synchronized.
- The Phase 12 controlled-shift validation result should not be interpreted as 100% real-world deepfake detection accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directory to path for imports
import sys
from pathlib import Path as _Path

_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.inference import SyncGuardPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpoint paths (relative to repo root)
CHECKPOINT_DIR = _repo_root / "outputs" / "runs"
AUDIO_ENCODER_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "audio_encoder.pt"
VISUAL_ENCODER_PATH = CHECKPOINT_DIR / "deepfake-transformer-final-20260908-210034" / "checkpoints" / "visual_encoder.pt"
SYNC_MODEL_PATH = CHECKPOINT_DIR / "sync-phase12-lambda01-20260913-115101" / "checkpoints" / "best.pt"
SYNC_CONFIG_PATH = _repo_root / "configs" / "av_align_lambda01.yaml"
SPOOF_HEAD_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "best.pt"

# Global predictor instance
_predictor: SyncGuardPredictor | None = None
_landmarker: Any = None


def get_predictor() -> SyncGuardPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        logger.info("Initializing SyncGuardPredictor...")
        try:
            _predictor = SyncGuardPredictor(
                audio_encoder_path=AUDIO_ENCODER_PATH,
                visual_encoder_path=VISUAL_ENCODER_PATH,
                sync_model_path=SYNC_MODEL_PATH,
                sync_config_path=SYNC_CONFIG_PATH,
                spoof_head_checkpoint=SPOOF_HEAD_PATH,
                device="auto",
            )
            logger.info(f"Predictor initialized on device: {_predictor.device}")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise RuntimeError(f"Failed to initialize predictor: {e}")
    return _predictor


def get_landmarker():
    """Get or create the MediaPipe landmarker instance."""
    global _landmarker
    if _landmarker is None:
        try:
            from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker
            model_path = ensure_model(DEFAULT_MODEL)
            _landmarker = make_landmarker(model_path)
            logger.info("MediaPipe landmarker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize landmarker: {e}")
            raise RuntimeError(f"Failed to initialize landmarker: {e}")
    return _landmarker


def process_audio_only(audio_file):
    """Process audio file for spoof detection."""
    if audio_file is None:
        return None, "Please upload an audio file first."
    
    try:
        predictor = get_predictor()
        result = predictor.predict_audio(audio_file)
        
        # Format result
        label = "BONAFIDE (REAL)" if result.predicted_label == "bonafide" else "SYNTHETIC (SPOOF)"
        spoof_pct = result.spoof_probability * 100
        bonafide_pct = result.bonafide_probability * 100
        confidence_pct = result.confidence * 100
        
        # Create result card
        result_html = f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 2px solid #0f3460;">
            <h2 style="color: #e94560; margin: 0 0 15px 0; font-size: 24px;">AUDIO-ONLY RESULT</h2>
            <div style="margin-bottom: 15px;">
                <span style="color: #ccc; font-size: 14px;">Prediction:</span>
                <div style="color: #4cc9f0; font-size: 28px; font-weight: bold; margin-top: 5px;">{label}</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <div>
                    <span style="color: #ccc; font-size: 12px;">Spoof Probability:</span>
                    <div style="color: #e94560; font-size: 20px; font-weight: bold;">{spoof_pct:.2f}%</div>
                </div>
                <div>
                    <span style="color: #ccc; font-size: 12px;">Bonafide Probability:</span>
                    <div style="color: #4cc9f0; font-size: 20px; font-weight: bold;">{bonafide_pct:.2f}%</div>
                </div>
            </div>
            <div>
                <span style="color: #ccc; font-size: 12px;">Confidence:</span>
                <div style="color: #f72585; font-size: 24px; font-weight: bold;">{confidence_pct:.2f}%</div>
            </div>
        </div>
        """
        
        explanation = """
        <div style="padding: 15px; border-radius: 8px; background: #0f3460; margin-top: 15px;">
            <p style="color: #ccc; margin: 0; font-size: 13px;">
                <strong>Note:</strong> Audio-only mode evaluates whether the speech signal is likely bonafide or spoofed/synthetic. 
                This is based on acoustic features learned from the ASVspoof dataset.
            </p>
        </div>
        """
        
        return result_html + explanation, None
        
    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        logger.error(error_msg)
        return None, f"❌ {error_msg}"
    except ValueError as e:
        error_msg = f"Processing error: {e}"
        logger.error(error_msg)
        return None, f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg)
        return None, f"❌ {error_msg}"


def process_audio_visual(video_file, audio_file, landmarks_file):
    """Process video file for AV synchronization analysis."""
    if video_file is None:
        return None, None, "Please upload a video file first."
    
    try:
        predictor = get_predictor()
        
        # Handle optional inputs
        audio_path = audio_file if audio_file else None
        landmarks_path = landmarks_file if landmarks_file else None
        landmarker = get_landmarker() if landmarks_path is None else None
        
        result = predictor.predict_audio_visual(
            video_path=video_file,
            audio_path=audio_path,
            landmarks_path=landmarks_path,
            landmarker=landmarker,
        )
        
        # Format result
        label = "SYNCHRONIZED" if result.predicted_label == "sync" else "DESYNCHRONIZED"
        sync_pct = result.sync_probability * 100
        desync_pct = result.desync_probability * 100
        aggregate_pct = result.aggregate_sync_score * 100
        
        # Create result card
        result_html = f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 2px solid #0f3460;">
            <h2 style="color: #e94560; margin: 0 0 15px 0; font-size: 24px;">AUDIO-VISUAL RESULT</h2>
            <div style="margin-bottom: 15px;">
                <span style="color: #ccc; font-size: 14px;">Prediction:</span>
                <div style="color: #4cc9f0; font-size: 28px; font-weight: bold; margin-top: 5px;">{label}</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <div>
                    <span style="color: #ccc; font-size: 12px;">Sync Probability:</span>
                    <div style="color: #4cc9f0; font-size: 20px; font-weight: bold;">{sync_pct:.2f}%</div>
                </div>
                <div>
                    <span style="color: #ccc; font-size: 12px;">Desync Probability:</span>
                    <div style="color: #e94560; font-size: 20px; font-weight: bold;">{desync_pct:.2f}%</div>
                </div>
            </div>
            <div>
                <span style="color: #ccc; font-size: 12px;">Aggregate Sync Score:</span>
                <div style="color: #f72585; font-size: 24px; font-weight: bold;">{aggregate_pct:.2f}%</div>
            </div>
        </div>
        """
        
        # Create timeline plot
        timeline_plot = create_timeline_plot(result)
        
        explanation = """
        <div style="padding: 15px; border-radius: 8px; background: #0f3460; margin-top: 15px;">
            <p style="color: #ccc; margin: 0; font-size: 13px;">
                <strong>Note:</strong> AV mode detects temporal synchronization inconsistencies between audio and visual streams. 
                It does not directly detect content manipulation. A synchronized manipulated video can still be a deepfake.
            </p>
        </div>
        """
        
        return result_html + explanation, timeline_plot, None
        
    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        logger.error(error_msg)
        return None, None, f"❌ {error_msg}"
    except ValueError as e:
        error_msg = f"Processing error: {e}"
        logger.error(error_msg)
        return None, None, f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg)
        return None, None, f"❌ {error_msg}"


def create_timeline_plot(result):
    """Create a timeline plot of per-window sync scores."""
    if result.per_window_sync_scores is None or result.timing_metadata is None:
        return None
    
    scores = result.per_window_sync_scores
    metadata = result.timing_metadata
    
    # Create time axis based on metadata
    fps = metadata.get("fps", 25.0)
    num_frames = metadata.get("num_frames", len(scores))
    time_points = np.arange(len(scores)) / fps
    
    # Create plot
    plt.figure(figsize=(10, 4), facecolor='#1a1a2e')
    ax = plt.gca()
    ax.set_facecolor('#16213e')
    
    # Plot sync scores
    colors = ['#4cc9f0' if s >= 0.5 else '#e94560' for s in scores]
    bars = ax.bar(time_points, scores, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Add threshold line
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='#ccc', fontsize=12)
    ax.set_ylabel('Sync Probability', color='#ccc', fontsize=12)
    ax.set_title('Synchronization Timeline', color='#e94560', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', colors='#ccc')
    ax.tick_params(axis='y', colors='#ccc')
    ax.spines['bottom'].set_color('#0f3460')
    ax.spines['top'].set_color('#0f3460')
    ax.spines['left'].set_color('#0f3460')
    ax.spines['right'].set_color('#0f3460')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.2, color='#ccc')
    
    plt.tight_layout()
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name, facecolor='#1a1a2e', dpi=100)
        temp_path = f.name
    
    plt.close()
    return temp_path


def create_architecture_diagram():
    """Create HTML architecture diagram."""
    return """
    <div style="padding: 20px; border-radius: 10px; background: #1a1a2e; border: 1px solid #0f3460;">
        <h3 style="color: #e94560; margin-top: 0;">How SyncGuard Works</h3>
        
        <div style="margin: 20px 0;">
            <h4 style="color: #4cc9f0; margin-bottom: 10px;">Audio-Only Mode:</h4>
            <div style="color: #ccc; font-size: 13px; line-height: 1.6;">
                Audio → Log-Mel Spectrogram → CNN + Transformer Audio Encoder → Spoof Head → REAL / SYNTHETIC
            </div>
        </div>
        
        <div style="margin: 20px 0;">
            <h4 style="color: #4cc9f0; margin-bottom: 10px;">Audio-Visual Mode:</h4>
            <div style="color: #ccc; font-size: 13px; line-height: 1.6;">
                <div style="margin-bottom: 8px;">
                    <strong>Video:</strong> Face/Mouth Landmarks → Visual Transformer → Visual Tokens
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Audio:</strong> CNN + Transformer → Audio Tokens
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Fusion:</strong> Temporal Alignment → Bidirectional Cross-Attention → Sync Head → SYNC / DESYNC
                </div>
            </div>
        </div>
    </div>
    """


def create_model_info():
    """Create model information panel."""
    try:
        predictor = get_predictor()
        device_str = "CUDA" if predictor.device.type == "cuda" else "CPU"
        audio_token_sec = predictor.audio_token_seconds
    except Exception:
        device_str = "Unknown"
        audio_token_sec = 0.01
    
    return f"""
    <div style="padding: 20px; border-radius: 10px; background: #1a1a2e; border: 1px solid #0f3460;">
        <h3 style="color: #e94560; margin-top: 0;">Model Information</h3>
        
        <div style="margin: 15px 0;">
            <div style="color: #ccc; font-size: 13px; margin-bottom: 8px;">
                <strong>Audio Encoder:</strong> Phase 5 CNN + Transformer
            </div>
            <div style="color: #ccc; font-size: 13px; margin-bottom: 8px;">
                <strong>Visual Encoder:</strong> Phase 8 Landmark Transformer
            </div>
            <div style="color: #ccc; font-size: 13px; margin-bottom: 8px;">
                <strong>AV Sync:</strong> Phase 12 λ=0.1 model
            </div>
            <div style="color: #ccc; font-size: 13px; margin-bottom: 8px;">
                <strong>Audio Token Timing:</strong> {audio_token_sec:.3f} seconds
            </div>
            <div style="color: #ccc; font-size: 13px; margin-bottom: 8px;">
                <strong>Device:</strong> {device_str}
            </div>
        </div>
        
        <div style="padding: 10px; border-radius: 5px; background: #0f3460; margin-top: 15px;">
            <p style="color: #f72585; margin: 0; font-size: 12px; font-weight: bold;">
                ⚠️ IMPORTANT LIMITATION
            </p>
            <p style="color: #ccc; margin: 5px 0 0 0; font-size: 11px;">
                The AV synchronization model evaluates temporal correspondence between audio and visual streams. 
                It is not a universal deepfake classifier. Manipulated content can remain synchronized.
                The Phase 12 controlled-shift validation result should not be interpreted as 100% real-world deepfake detection accuracy.
            </p>
        </div>
    </div>
    """


def create_ui():
    """Create the Gradio UI."""
    
    # Custom CSS for dark theme
    custom_css = """
    .gradio-container {
        background: #0f0f1a !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #e94560 0%, #f72585 100%) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #f72585 0%, #e94560 100%) !important;
    }
    """
    
    with gr.Blocks(
        title="SyncGuard - Multimodal Deepfake Detection",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown(
            """
            # SYNCGUARD
            ## Multimodal Deepfake Detection
            
            **Audio spoof detection + audio-visual synchronization analysis**
            """
        )
        
        with gr.Tabs():
            
            # Tab 1: Audio-Only Detection
            with gr.Tab("Audio-Only Detection"):
                gr.Markdown("### Audio-Only Spoof Detection")
                gr.Markdown("Upload an audio file to analyze whether it's likely bonafide or synthetic/spoofed.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        audio_input = gr.Audio(
                            label="Upload Audio",
                            type="filepath"
                        )
                        analyze_audio_btn = gr.Button("Analyze Audio", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        audio_result = gr.HTML(label="Result")
                        audio_error = gr.Textbox(label="Status", visible=False)
                
                analyze_audio_btn.click(
                    fn=process_audio_only,
                    inputs=[audio_input],
                    outputs=[audio_result, audio_error]
                )
            
            # Tab 2: Audio-Visual Sync
            with gr.Tab("Audio-Visual Sync"):
                gr.Markdown("### Audio-Visual Synchronization Analysis")
                gr.Markdown("Upload a video to analyze temporal synchronization between audio and visual streams.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        video_input = gr.Video(label="Upload Video")
                        
                        with gr.Accordion("Advanced Options", open=False):
                            audio_input_av = gr.Audio(label="Separate Audio (Optional)", type="filepath")
                            landmarks_input = gr.File(label="Precomputed Landmarks (Optional)", file_types=[".npz"])
                        
                        analyze_av_btn = gr.Button("Analyze Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        av_result = gr.HTML(label="Result")
                        timeline_plot = gr.Image(label="Sync Timeline")
                        av_error = gr.Textbox(label="Status", visible=False)
                
                analyze_av_btn.click(
                    fn=process_audio_visual,
                    inputs=[video_input, audio_input_av, landmarks_input],
                    outputs=[av_result, timeline_plot, av_error]
                )
            
            # Tab 3: Information
            with gr.Tab("How It Works"):
                gr.HTML(create_architecture_diagram())
                gr.HTML(create_model_info())
        
        # Footer
        gr.Markdown(
            """
            ---
            **Note:** This is a research demonstration. Results should not be used for security-critical decisions without further validation.
            """
        )
    
    return demo


def main():
    """Main entry point for the demo application."""
    logger.info("Starting SyncGuard Demo UI...")
    
    # Initialize predictor on startup
    try:
        get_predictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        print(f"Error: Failed to initialize predictor: {e}")
        print("Please ensure all checkpoint files exist in the expected locations.")
        return
    
    # Create and launch UI
    demo = create_ui()
    
    logger.info("Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
