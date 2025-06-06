"""
TTS optimization utilities for faster Chatterbox synthesis
"""

def get_optimized_chatterbox_settings():
    """Get optimized settings for faster Chatterbox TTS synthesis"""
    return {
        # Reduce quality slightly for speed
        'voice_temperature': 0.5,  # Lower = more deterministic and faster
        'voice_cfg_weight': 0.3,   # Lower = less guidance, faster generation
        'voice_exaggeration': 0.3, # Lower = less emotional processing
        
        # Optimize generation parameters
        'max_length': 1000,        # Shorter max length
        'guidance_scale': 3.0,     # Lower guidance scale (default often 7.5)
        'num_inference_steps': 15, # Reduce from default 20-25 steps
        
        # Memory optimizations
        'enable_attention_slicing': True,
        'enable_vae_slicing': True,
        'enable_cpu_offload': False,  # Keep on GPU for speed
    }

def apply_tts_optimizations(tts_wrapper):
    """Apply performance optimizations to Chatterbox TTS"""
    optimized_settings = get_optimized_chatterbox_settings()
    
    # Apply settings if the wrapper supports them
    for setting, value in optimized_settings.items():
        if hasattr(tts_wrapper, setting):
            setattr(tts_wrapper, setting, value)
        elif hasattr(tts_wrapper, 'tts') and hasattr(tts_wrapper.tts, setting):
            setattr(tts_wrapper.tts, setting, value)
    
    print("🚀 Applied TTS optimizations for faster synthesis")
    return tts_wrapper

def get_speed_optimized_voice_settings():
    """Get voice settings optimized for speed over quality"""
    return {
        'voice_temperature': 0.4,     # Even faster
        'voice_cfg_weight': 0.2,      # Minimal guidance
        'voice_exaggeration': 0.2,    # Minimal emotion processing
    }