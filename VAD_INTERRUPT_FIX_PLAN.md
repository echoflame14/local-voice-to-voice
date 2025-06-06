# VAD Interrupt and Response Logging Fix Plan

## Problem Analysis

### Issue 1: VAD Over-Sensitivity
The VAD is triggering on ANY audio, including:
- Assistant's own voice bleeding through speakers
- Background noise
- Mic feedback
- Random sounds

This causes immediate interruption of the assistant's response.

### Issue 2: Incorrect Response Logging
The system logs the ENTIRE LLM response as "spoken" even when:
- The response was interrupted immediately
- No audio was actually synthesized
- The user never heard most of the content

Example from logs:
- Interrupt happens at 7.1s
- Claims "63/63 sentences completed" 
- Logs massive multi-paragraph response that was never spoken

## Root Causes

### 1. VAD Sensitivity Issues
```python
# Current issue in input_manager.py
if is_speech:  # This triggers on ANY detected speech
    self.consecutive_speech_frames += 1
    # Triggers interrupt immediately without validation
```

### 2. Response Logging Issues
```python
# Current issue in voice_assistant.py
# The 'spoken_response' list is being populated with ALL chunks
# Not tracking which chunks were ACTUALLY synthesized and played
spoken_response.append(tts_chunk)  # This happens even if synthesis fails
```

## Solution Design

### Part 1: Fix VAD Over-Sensitivity

#### A. Add Grace Period for Interrupts
```python
class InputManager:
    def __init__(self):
        # Existing config
        self.interrupt_grace_period = 1.5  # seconds
        self.last_assistant_audio_time = 0
        self.min_interrupt_confidence = 0.7  # Require high confidence
        
    def process_audio_frame(self, frame):
        # Check if we're in grace period
        time_since_assistant = time.time() - self.last_assistant_audio_time
        if time_since_assistant < self.interrupt_grace_period:
            return None  # Ignore potential interrupts during grace period
```

#### B. Add Audio Level Threshold
```python
def is_valid_interrupt(self, audio_frame):
    """Validate interrupt with multiple checks"""
    # 1. Check audio level
    audio_level = np.abs(audio_frame).mean()
    if audio_level < self.min_audio_level:
        return False
        
    # 2. Check if different from assistant's audio
    if self.is_echo_or_feedback(audio_frame):
        return False
        
    # 3. Require sustained speech
    if self.consecutive_speech_frames < self.min_speech_frames:
        return False
        
    return True
```

#### C. Echo Cancellation Check
```python
def is_echo_or_feedback(self, audio_frame):
    """Check if audio is echo/feedback from assistant"""
    if hasattr(self, 'assistant_audio_buffer'):
        # Cross-correlation to detect similarity
        correlation = np.correlate(audio_frame, self.assistant_audio_buffer)
        if correlation.max() > self.echo_threshold:
            return True
    return False
```

### Part 2: Fix Response Logging

#### A. Track Synthesis State
```python
class VoiceAssistant:
    def __init__(self):
        self.synthesis_tracker = {
            'attempted_chunks': [],
            'synthesized_chunks': [],
            'played_chunks': [],
            'interrupted_at_chunk': None
        }
```

#### B. Update Synthesis Tracking
```python
def synthesize_chunk(self, chunk_text, chunk_index):
    """Track synthesis state accurately"""
    self.synthesis_tracker['attempted_chunks'].append({
        'index': chunk_index,
        'text': chunk_text,
        'attempted_at': time.time()
    })
    
    try:
        # Synthesis attempt
        audio, sample_rate = self.tts.synthesize(chunk_text)
        
        self.synthesis_tracker['synthesized_chunks'].append({
            'index': chunk_index,
            'text': chunk_text,
            'synthesized_at': time.time()
        })
        
        # Playback attempt
        if self.play_audio(audio, sample_rate):
            self.synthesis_tracker['played_chunks'].append({
                'index': chunk_index,
                'text': chunk_text,
                'played_at': time.time()
            })
            return True
            
    except InterruptedException:
        self.synthesis_tracker['interrupted_at_chunk'] = chunk_index
        return False
```

#### C. Accurate Response Logging
```python
def log_assistant_response(self):
    """Log only what was actually spoken"""
    # Get only the chunks that were fully played
    actual_spoken_text = ' '.join([
        chunk['text'] for chunk in self.synthesis_tracker['played_chunks']
    ])
    
    # Calculate actual metrics
    sentences_spoken = len([c for c in self.synthesis_tracker['played_chunks']])
    total_sentences = len(self.synthesis_tracker['attempted_chunks'])
    
    # Log with accurate metadata
    self.conversation_logger.log_assistant_message(
        actual_spoken_text,
        metadata={
            'sentences_spoken': sentences_spoken,
            'total_sentences': total_sentences,
            'interrupted': self.synthesis_tracker['interrupted_at_chunk'] is not None,
            'interrupt_time': self.playback_interrupted_at
        }
    )
```

### Part 3: Immediate Fixes

#### 1. Increase VAD Thresholds
```python
# In config.py
VAD_AGGRESSIVENESS = 3  # Maximum (was 1)
VAD_MIN_SPEECH_FRAMES = 10  # Require more frames (was 3)
VAD_INTERRUPT_GRACE_PERIOD = 2.0  # seconds
```

#### 2. Add Confidence Scoring
```python
def get_vad_confidence(self, audio_frame):
    """Get confidence score for speech detection"""
    # Use multiple indicators
    energy = np.abs(audio_frame).mean()
    zero_crossings = np.sum(np.diff(np.sign(audio_frame)) != 0)
    
    # Normalize and combine
    energy_score = min(energy / self.energy_threshold, 1.0)
    zc_score = min(zero_crossings / self.zc_threshold, 1.0)
    
    return (energy_score + zc_score) / 2
```

#### 3. Fix Spoken Response Tracking
```python
# In voice_assistant.py stream_response_with_tts()
spoken_chunks = []  # Track only successfully played chunks

# In synthesis loop
if playback_successful:
    spoken_chunks.append(tts_chunk)
    
# At end of response
actual_spoken = ' '.join(spoken_chunks)
self.log_response(actual_spoken, full_response_text)
```

## Implementation Priority

1. **Critical** - Fix response logging (prevents incorrect conversation history)
2. **High** - Add VAD grace period (prevents most false interrupts)  
3. **High** - Track synthesis state accurately
4. **Medium** - Add echo cancellation
5. **Medium** - Implement confidence scoring
6. **Low** - Advanced audio analysis

## Testing Plan

1. Test with assistant speaking - should not self-interrupt
2. Test with background noise - should not trigger interrupt
3. Test with actual user speech - should interrupt appropriately
4. Verify logged responses match what was actually played
5. Test interrupt timing and grace periods

## Expected Outcomes

1. VAD will not trigger on assistant's own voice
2. Background noise won't cause interrupts
3. Only actual user speech will interrupt
4. Conversation logs will accurately reflect what was spoken
5. Interrupt metadata will be accurate