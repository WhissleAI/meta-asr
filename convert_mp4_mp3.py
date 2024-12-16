import os
from typing import Generator, Optional
from moviepy import VideoFileClip
from pydub import AudioSegment
from io import BytesIO

class MP4AudioChunkConverter:

    def __init__(self, file_path: str, chunk_duration: int = 5000, temp_audio_path: Optional[str] = None):
    
        self.file_path = file_path
        self.chunk_duration = chunk_duration
        self.temp_audio_path = temp_audio_path or "temp_audio.wav"
    
    def convert_to_chunks(self) -> Generator[BytesIO, None, None]:

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        
        try:
            video = VideoFileClip(self.file_path)
            audio = video.audio
            
            audio.write_audiofile(self.temp_audio_path, codec="pcm_s16le")
            

            audio_segment = AudioSegment.from_file(self.temp_audio_path, format="wav")
            
            
            for i in range(0, len(audio_segment), self.chunk_duration):
                chunk = audio_segment[i:i + self.chunk_duration]
                chunk_buffer = BytesIO()
                chunk.export(chunk_buffer, format="mp3")
                chunk_buffer.seek(0)
                yield chunk_buffer
        
        except Exception as e:
            raise Exception(f"Error converting audio: {e}")
        
        finally:

            if os.path.exists(self.temp_audio_path):
                os.remove(self.temp_audio_path)
          
            if 'video' in locals():
                video.close()
    
    def __repr__(self) -> str:
        return f"MP4AudioChunkConverter(file_path='{self.file_path}', chunk_duration={self.chunk_duration})"