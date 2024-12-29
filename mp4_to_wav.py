import os
from typing import Optional
from moviepy import VideoFileClip

class MP4ToWavConverter:
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        self.input_path = input_path
        self.output_dir = "converted_audio"
        self.output_path = output_path or self._generate_output_path()
        
    def _generate_output_path(self) -> str:

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
        base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
        return os.path.join(self.output_dir, f"{base_filename}.wav")
    
    def convert(self) -> str:

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        try:
            video = VideoFileClip(self.input_path)
            audio = video.audio
            audio.write_audiofile(
                self.output_path,
                codec='pcm_s16le', 
                fps=44100,  
                nbytes=2, 
                buffersize=2000,
                write_logfile=False
            )
            return self.output_path
            
        except Exception as e:
            raise Exception(f"Error converting video to audio: {e}")
            
        finally:
            if 'video' in locals():
                video.close()
                
    def __repr__(self) -> str:
        return f"MP4ToWavConverter(input_path='{self.input_path}', output_path='{self.output_path}')"
    
    
converter = MP4ToWavConverter("A one minute TEDx Talk for the digital age _ Woody Roseland _ TEDxMileHigh.mp4")
wav_path = converter.convert()
print(wav_path)