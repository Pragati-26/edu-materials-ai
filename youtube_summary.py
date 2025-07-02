from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yt_dlp
import whisper
import os
import uuid
import subprocess

router = APIRouter()
model = whisper.load_model("base")  # or "small" / "medium"

class VideoURL(BaseModel):
    url: str

@router.post("/transcribe-youtube/")
def transcribe_youtube_video(data: VideoURL):
    base_name = f"temp-{uuid.uuid4().hex}"
    downloaded_file = f"{base_name}.webm"
    audio_file = f"{base_name}.mp3"

    try:
        print("🎯 Received request for:", data.url)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': downloaded_file,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("⬇️  Downloading...")
            ydl.download([data.url])
        print("✅ Download completed")

        # ✅ Convert to mp3 using ffmpeg
        convert_cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-i", downloaded_file,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "192k",
            audio_file
        ]
        subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 🔍 Check if audio file is valid
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1024:
            raise HTTPException(status_code=500, detail="Downloaded audio is too small or corrupt")

        print("🎧 Transcribing...")
        result = model.transcribe(audio_file)
        print("✅ Transcription done")

        return {
            "message": "Transcription successful",
            "transcript": result['text']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        for f in [downloaded_file, audio_file]:
            if os.path.exists(f):
                os.remove(f)
