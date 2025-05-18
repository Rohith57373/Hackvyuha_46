# SecureView CCTV Monitoring System

A Flask-based WebRTC video streaming application that combines CCTV monitoring features with real-time video streaming capabilities.

## Features

- Beautiful CCTV monitoring dashboard UI
- Real-time WebRTC video streaming
- Room-based video conferencing
- Camera status monitoring
- Responsive design

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd flask_cctv
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. To start a WebRTC stream, click on the "Start WebRTC Stream" button.

4. Share the generated room URL with others to join your video stream.

## System Requirements

- Python 3.7+
- Modern web browser with WebRTC support (Chrome, Firefox, Edge, etc.)
- Camera and microphone access

## License

MIT 