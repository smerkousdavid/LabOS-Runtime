// Device selection and dynamic data
const cameraSelect = document.getElementById('cameraSelect');
const pathDisplay = document.getElementById('pathDisplay');
const streamPaths = document.getElementById('streamPaths');
const connectionStatus = document.getElementById('connectionStatus');
let selectedCamera = null;
let availableCameras = [];

// TTS models data
let availableTtsModels = [];
let selectedTtsModel = null;

// Tab management
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        const tabName = button.getAttribute('data-tab');
        
        // Remove active class from all buttons and contents
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked button and corresponding content
        button.classList.add('active');
        document.getElementById(tabName).classList.add('active');
    });
});

// Check connection and load devices on page load
window.addEventListener('load', () => {
    checkStatus();
    loadCameras();
    loadTtsModels();
});

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.xr_service_connected) {
            connectionStatus.textContent = 'Connected to XR Service';
            connectionStatus.className = 'connection-status connected';
        } else {
            connectionStatus.textContent = 'Disconnected from XR Service';
            connectionStatus.className = 'connection-status disconnected';
        }
    } catch (error) {
        connectionStatus.textContent = 'Error checking connection';
        connectionStatus.className = 'connection-status disconnected';
    }
}

async function loadCameras() {
    try {
        const response = await fetch('/api/cameras');
        const data = await response.json();
        
        if (data.success && data.cameras) {
            availableCameras = data.cameras;
            
            // Populate dropdown
            cameraSelect.innerHTML = '';
            availableCameras.forEach(camera => {
                const option = document.createElement('option');
                option.value = camera.id;
                option.textContent = `Device ${camera.id}`;
                cameraSelect.appendChild(option);
            });
            
            // Select first device by default
            if (availableCameras.length > 0) {
                cameraSelect.value = availableCameras[0].id;
                selectedCamera = availableCameras[0];
                updatePathDisplay();
            }
        }
    } catch (error) {
        console.error('Failed to load cameras:', error);
    }
}

async function loadTtsModels() {
    try {
        const response = await fetch('/api/tts_models');
        const data = await response.json();
        
        if (data.success && data.models) {
            availableTtsModels = data.models;
            
            // Populate TTS model dropdown
            const ttsModelSelect = document.getElementById('ttsModel');
            ttsModelSelect.innerHTML = '';
            
            availableTtsModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                ttsModelSelect.appendChild(option);
            });
            
            // Select first model by default
            if (availableTtsModels.length > 0) {
                ttsModelSelect.value = availableTtsModels[0].id;
                selectedTtsModel = availableTtsModels[0];
                updateTtsVoices();
            }
        }
    } catch (error) {
        console.error('Failed to load TTS models:', error);
        // Fallback to hardcoded options if API fails
        const ttsModelSelect = document.getElementById('ttsModel');
        ttsModelSelect.innerHTML = `
            <option value="riva">Riva TTS</option>
            <option value="qwen-tts-flash">Qwen TTS Flash</option>
        `;
    }
}

// Camera selection handler
cameraSelect.addEventListener('change', () => {
    const cameraId = cameraSelect.value;
    selectedCamera = availableCameras.find(c => c.id === parseInt(cameraId));
    updatePathDisplay();
});

// TTS model selection handler
document.getElementById('ttsModel').addEventListener('change', () => {
    const modelId = document.getElementById('ttsModel').value;
    selectedTtsModel = availableTtsModels.find(m => m.id === modelId);
    updateTtsVoices();
});

function updateTtsVoices() {
    if (!selectedTtsModel) return;
    
    const ttsVoiceSelect = document.getElementById('ttsVoice');
    ttsVoiceSelect.innerHTML = '';
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = 'default';
    defaultOption.textContent = 'Default';
    ttsVoiceSelect.appendChild(defaultOption);
    
    // Add available voices for this model
    if (selectedTtsModel.voices && selectedTtsModel.voices.length > 0) {
        selectedTtsModel.voices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            option.textContent = voice;
            ttsVoiceSelect.appendChild(option);
        });
    }
    
    // Select default voice
    ttsVoiceSelect.value = selectedTtsModel.default_voice || 'default';
}

function updatePathDisplay() {
    if (!selectedCamera) return;
    
    pathDisplay.style.display = 'block';
    
    streamPaths.innerHTML = `
        <div class="path-item">
            <span class="path-label">Device:</span>
            <span class="path-value">Device ${selectedCamera.id}</span>
        </div>
        <div class="path-item">
            <span class="path-label">Device App Port:</span>
            <span class="path-value">${selectedCamera.appPort}</span>
        </div>
        <div class="path-item">
            <span class="path-label">RTSP Path (Merged - Video+Audio):</span>
            <span class="path-value">${selectedCamera.streams.merged}</span>
        </div>
        <div class="path-item">
            <span class="path-label">RTSP Path (Video Only):</span>
            <span class="path-value">${selectedCamera.streams.video}</span>
        </div>
        <div class="path-item">
            <span class="path-label">RTSP Path (Audio Only):</span>
            <span class="path-value">${selectedCamera.streams.audio}</span>
        </div>
        <div class="path-item">
            <span class="path-label">TTS Output Stream:</span>
            <span class="path-value">${selectedCamera.streams.tts}</span>
        </div>
    `;
    
    // Update TTS path info
    document.getElementById('ttsPaths').textContent = selectedCamera.streams.tts;
    document.getElementById('ttsPathInfo').style.display = 'block';
}

// Refresh connection status every 10 seconds
setInterval(checkStatus, 10000);

// ========== MESSAGE FORM ==========
const messageForm = document.getElementById('messageForm');
const messageStatus = document.getElementById('messageStatus');
const sendMessageBtn = document.getElementById('sendMessageBtn');

messageForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedCamera) {
        showStatus('Please select a camera', 'error', 'messageStatus');
        return;
    }
    
    const messageType = document.getElementById('messageType').value;
    const payload = document.getElementById('payload').value;
    
    sendMessageBtn.disabled = true;
    const originalText = sendMessageBtn.textContent;
    sendMessageBtn.textContent = 'Sending...';
    
    try {
        const response = await fetch(`http://${window.location.hostname}:${selectedCamera.port}/api/send_message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message_type: messageType,
                payload: payload
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Message sent successfully!', 'success', 'messageStatus');
        } else {
            showStatus(`Error: ${data.error || data.message}`, 'error', 'messageStatus');
        }
    } catch (error) {
        showStatus(`Network error: ${error.message}`, 'error', 'messageStatus');
    } finally {
        sendMessageBtn.disabled = false;
        sendMessageBtn.textContent = originalText;
        checkStatus();
    }
});

// ========== AUDIO FORM ==========
const audioForm = document.getElementById('audioForm');
const audioStatus = document.getElementById('audioStatus');
const sendAudioBtn = document.getElementById('sendAudioBtn');
const audioFileInput = document.getElementById('audioFile');
const sampleRateInput = document.getElementById('sampleRate');
const audioMethodSelect = document.getElementById('audioMethod');
const chunkDurationGroup = document.getElementById('chunkDurationGroup');

audioMethodSelect.addEventListener('change', () => {
    if (audioMethodSelect.value === 'streaming') {
        chunkDurationGroup.style.display = 'block';
    } else {
        chunkDurationGroup.style.display = 'none';
    }
});

audioFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.wav')) return;
    
    try {
        const arrayBuffer = await file.arrayBuffer();
        const dataView = new DataView(arrayBuffer);
        const riff = String.fromCharCode(dataView.getUint8(0), dataView.getUint8(1), 
                                         dataView.getUint8(2), dataView.getUint8(3));
        if (riff === 'RIFF') {
            const detectedSampleRate = dataView.getUint32(24, true);
            sampleRateInput.value = detectedSampleRate;
            showStatus(`Detected WAV file: ${detectedSampleRate} Hz`, 'info', 'audioStatus');
            setTimeout(() => { audioStatus.style.display = 'none'; }, 2000);
        }
    } catch (error) {
        console.log('Could not auto-detect sample rate:', error);
    }
});

audioForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedCamera) {
        showStatus('Please select a camera', 'error', 'audioStatus');
        return;
    }
    
    const audioFile = audioFileInput.files[0];
    const sampleRate = sampleRateInput.value;
    const audioMethod = audioMethodSelect.value;
    
    if (!audioFile) {
        showStatus('Please select an audio file', 'error', 'audioStatus');
        return;
    }
    
    sendAudioBtn.disabled = true;
    const originalText = sendAudioBtn.textContent;
    sendAudioBtn.textContent = audioMethod === 'streaming' ? 'Streaming Audio...' : 'Sending Audio...';
    
    try {
        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('sample_rate', sampleRate);
        formData.append('method', audioMethod);
        
        if (audioMethod === 'streaming') {
            const chunkDuration = document.getElementById('chunkDuration').value;
            formData.append('chunk_duration', chunkDuration);
        }
        
        const response = await fetch(`http://${window.location.hostname}:${selectedCamera.port}/api/send_audio`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            let message = `✅ ${data.message}`;
            if (data.data.detected_rate && data.data.user_specified_rate && 
                Math.abs(data.data.detected_rate - data.data.user_specified_rate) > 100) {
                message += `\n⚠️ Note: WAV file sample rate (${data.data.detected_rate} Hz) was used`;
            }
            showStatus(message, 'success', 'audioStatus');
        } else {
            showStatus(`❌ Error: ${data.error}`, 'error', 'audioStatus');
        }
    } catch (error) {
        showStatus(`❌ Network error: ${error.message}`, 'error', 'audioStatus');
    } finally {
        sendAudioBtn.disabled = false;
        sendAudioBtn.textContent = originalText;
        checkStatus();
    }
});

// ========== TTS FORM ==========
const ttsForm = document.getElementById('ttsForm');
const ttsStatus = document.getElementById('ttsStatus');
const sendTtsBtn = document.getElementById('sendTtsBtn');

ttsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedCamera) {
        showStatus('Please select a camera', 'error', 'ttsStatus');
        return;
    }
    
    const ttsText = document.getElementById('ttsText').value;
    const ttsModel = document.getElementById('ttsModel').value;
    const ttsLanguage = document.getElementById('ttsLanguage').value;
    const ttsVoice = document.getElementById('ttsVoice').value;
    
    sendTtsBtn.disabled = true;
    sendTtsBtn.textContent = 'Generating...';
    
    try {
        const response = await fetch(`http://${window.location.hostname}:${selectedCamera.port}/api/send_tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: ttsText,
                model: ttsModel,
                language: ttsLanguage,
                voice: ttsVoice,
                camera_id: selectedCamera.id
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const ttsPath = `rtsp://localhost:8554/NB_${String(selectedCamera.id).padStart(4, '0')}_RX_TTS`;
            showStatus(`✅ TTS generated and streaming to: ${ttsPath}`, 'success', 'ttsStatus');
            document.getElementById('ttsText').value = ''; // Clear text
        } else {
            showStatus(`❌ Error: ${data.error || data.message}`, 'error', 'ttsStatus');
        }
    } catch (error) {
        showStatus(`❌ Network error: ${error.message}`, 'error', 'ttsStatus');
    } finally {
        sendTtsBtn.disabled = false;
        sendTtsBtn.textContent = 'Generate & Send TTS';
        checkStatus();
    }
});

// ========== STATUS HELPER ==========
function showStatus(message, type, statusElementId) {
    const statusDiv = document.getElementById(statusElementId);
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 4000);
    }
}
