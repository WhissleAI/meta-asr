import os
from pydub import AudioSegment
from typing import List, Dict
from pathlib import Path

class BatchAudioChunker:
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str = "chunks",
                 chunk_duration: int = 20000):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def _chunk_single_file(self, audio_path: str) -> List[str]:
        chunk_paths = []
        
        try:
            audio = AudioSegment.from_file(audio_path)
            
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            num_chunks = len(audio) // self.chunk_duration + (1 if len(audio) % self.chunk_duration else 0)

            for i in range(num_chunks):
                start = i * self.chunk_duration
                end = min((i + 1) * self.chunk_duration, len(audio))
                
                chunk = audio[start:end]
                chunk_path = os.path.join(self.output_dir, f"{base_name}_{i+1}.wav")
                
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
            
            return chunk_paths
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return []
    
    def process_directory(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        results = {}
        
  
        for filename in os.listdir(self.input_dir):
            if any(filename.lower().endswith(fmt) for fmt in self.supported_formats):
                file_path = os.path.join(self.input_dir, filename)
                chunk_paths = self._chunk_single_file(file_path)
                
                if chunk_paths: 
                    base_name = os.path.splitext(filename)[0]
                    results[base_name] = chunk_paths
        
        return results
    
    def get_processed_files(self) -> List[str]:
        """Get list of all processed files in output directory"""
        if not os.path.exists(self.output_dir):
            return []
        return [f for f in os.listdir(self.output_dir) 
                if f.endswith('.wav')]
    
    def __repr__(self) -> str:
        return (f"BatchAudioChunker(input_dir='{self.input_dir}', "
                f"output_dir='{self.output_dir}', "
                f"chunk_duration={self.chunk_duration})")


chunker = BatchAudioChunker(
    input_dir="converted_audio",     
    output_dir="chunks",          
    chunk_duration=20000         
)
results = chunker.process_directory()