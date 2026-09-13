# SyncGuard Demo UI

Phase 13B demonstration interface for SyncGuard multimodal deepfake detection.

## Installation

Ensure all dependencies are installed:

```bash
pip install -r ../requirements.txt
```

Note: Gradio should be added to requirements.txt if not already present.

## Launching the Demo

Run the demo application:

```bash
python app/app.py
```

The UI will be available at `http://localhost:7860`

## Usage

### Audio-Only Mode

1. Click the "Audio-Only Detection" tab
2. Upload an audio file (WAV, FLAC, etc.)
3. Click "Analyze Audio"
4. View the spoof/bonafide classification result

**What it detects:** Whether the speech signal is likely bonafide (real) or synthetic/spoofed based on acoustic features.

### Audio-Visual Mode

1. Click the "Audio-Visual Sync" tab
2. Upload a video file (MP4, etc.)
3. Optionally:
   - Upload a separate audio file if you want to use different audio
   - Upload precomputed landmarks (NPZ format) to skip MediaPipe extraction
4. Click "Analyze Video"
5. View the synchronization result and timeline

**What it detects:** Temporal synchronization inconsistencies between audio and visual streams.

### Information Tab

View:
- Architecture diagrams for both modes
- Model information (encoders, device, timing)
- Important limitations

## Important Limitations

### Audio-Only Mode
- Trained on ASVspoof dataset for synthetic speech detection
- Does not detect all types of audio manipulation
- Performance may vary on different recording conditions

### Audio-Visual Mode
- **This is NOT a universal deepfake detector**
- Specifically detects temporal misalignment between audio and video
- A synchronized manipulated video can still be a deepfake
- An audio-only manipulated clip can remain synchronized
- The Phase 12 controlled-shift validation (AUC=1.0) was on artificial temporal shifts, not real deepfake content
- This result should not be interpreted as 100% real-world deepfake detection accuracy

## Model Architecture

### Audio-Only Pipeline
```
Audio → Log-Mel Spectrogram → CNN + Transformer Audio Encoder → Spoof Head → REAL / SYNTHETIC
```

### Audio-Visual Pipeline
```
Video → Face/Mouth Landmarks → Visual Transformer → Visual Tokens
Audio → CNN + Transformer → Audio Tokens
Audio + Visual → Temporal Alignment → Bidirectional Cross-Attention → Sync Head → SYNC / DESYNC
```

## Checkpoints Used

- **Audio Encoder:** Phase 5 CNN + Transformer (`spoof-transformer-20260906-123646`)
- **Visual Encoder:** Phase 8 Landmark Transformer (`deepfake-transformer-final-20260908-210034`)
- **AV Sync Model:** Phase 12 λ=0.1 contrastive learning (`sync-phase12-lambda01-20260913-115101`)
- **Spoof Head:** Phase 5 spoof classifier (`spoof-transformer-20260906-123646`)

## Technical Details

- Audio sample rate: 16 kHz
- Audio token timing: 0.01 seconds
- Landmark format: MediaPipe 478-point face landmarks
- Device: Auto-selects CUDA if available, otherwise CPU
- Inference mode: `torch.inference_mode()` for efficiency

## Troubleshooting

### "Failed to initialize predictor"
- Ensure checkpoint files exist in `outputs/runs/`
- Check that `configs/av_align_lambda01.yaml` exists
- Verify PyTorch and dependencies are correctly installed

### "File not found" errors
- Ensure uploaded files are accessible
- Check file permissions
- Verify file format is supported

### CUDA not available
- The app will automatically fall back to CPU
- Performance will be slower but still functional

### MediaPipe errors
- Ensure MediaPipe is correctly installed
- Check that video contains detectable faces
- Try with a different video if landmark extraction fails

## Performance Notes

- Models are loaded once at startup
- Subsequent analyses reuse the same predictor instance
- First analysis may be slower due to model loading
- AV mode with landmark extraction is slower than audio-only mode

## Development

To run tests for the UI:

```bash
python -m pytest tests/test_app.py -v
```

## License

This is a research demonstration. Results should not be used for security-critical decisions without further validation.
