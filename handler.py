#!/usr/bin/env python3
"""
RunPod Serverless Handler for Vocal Removal
Processes vocal removal requests on-demand with GPU acceleration
"""

import runpod
import os
import json
import base64
import tempfile
import logging
import subprocess
import shutil
import glob
from datetime import datetime
import torch
import torchaudio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerlessVocalRemover:
    """Serverless vocal remover optimized for RunPod"""
    
    def __init__(self):
        self.temp_dir = "/tmp/vocal_removal"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Serverless worker using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            
        # Pre-warm models to reduce cold start time
        self.preload_models()
    
    def preload_models(self):
        """Pre-load models to reduce cold start time"""
        try:
            logger.info("Pre-loading Demucs models...")
            from demucs.pretrained import get_model
            
            # Load commonly used models
            models_to_preload = ['htdemucs_ft', 'htdemucs']
            for model_name in models_to_preload:
                try:
                    model = get_model(model_name)
                    model.to(self.device)
                    logger.info(f"Pre-loaded model: {model_name}")
                    # Clear memory after loading
                    del model
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Could not pre-load {model_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Model pre-loading failed: {e}")
    
    def decode_audio_file(self, audio_base64: str, filename: str) -> str:
        """Decode base64 audio data to file"""
        try:
            # Create temporary file
            file_path = os.path.join(self.temp_dir, f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
            
            # Decode and save
            audio_data = base64.b64decode(audio_base64)
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Decoded audio file: {file_path} ({len(audio_data)} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"Error decoding audio file: {e}")
            raise
    
    def encode_audio_file(self, file_path: str) -> str:
        """Encode audio file to base64"""
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
                encoded = base64.b64encode(audio_data).decode('utf-8')
            
            logger.info(f"Encoded result file: {len(encoded)} characters")
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding audio file: {e}")
            raise
    
    def remove_vocals_serverless(self, input_path: str, model: str = 'htdemucs_ft') -> str:
        """
        Remove vocals using Demucs on serverless GPU
        """
        try:
            logger.info(f"Starting serverless vocal removal: {model}")
            
            # Create output directory
            output_dir = os.path.join(self.temp_dir, f"demucs_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Optimized Demucs command for serverless
            command = [
                'python', '-m', 'demucs',
                '--mp3',  # Output format
                '--two-stems=vocals',  # Only vocals/accompaniment
                '-n', model,  # Model to use
                '-o', output_dir,  # Output directory
                '--device', self.device,  # GPU device
                '--shifts', '1',  # Fast processing
                '--overlap', '0.25',  # Balance speed/quality
                '--jobs', '1',  # Single job for GPU
                input_path  # Input file
            ]
            
            logger.info(f"Running Demucs: {' '.join(command)}")
            
            # Execute with timeout
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Demucs failed: {result.stderr}")
                logger.error(f"Demucs stdout: {result.stdout}")
                raise RuntimeError(f"Demucs processing failed: {result.stderr}")
            
            logger.info("Demucs processing completed successfully")
            
            # Find output file
            input_filename = os.path.splitext(os.path.basename(input_path))[0]
            no_vocals_pattern = os.path.join(output_dir, model, input_filename, "no_vocals.*")
            
            matching_files = glob.glob(no_vocals_pattern)
            if not matching_files:
                # Try alternative pattern
                no_vocals_pattern = os.path.join(output_dir, "**", "no_vocals.*")
                matching_files = glob.glob(no_vocals_pattern, recursive=True)
            
            if not matching_files:
                raise RuntimeError("Could not find Demucs output file")
            
            output_path = matching_files[0]
            logger.info(f"Found vocals-removed file: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in serverless vocal removal: {e}")
            raise
    
    def cleanup_files(self, *file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    logger.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up {file_path}: {e}")

# Global processor instance
processor = ServerlessVocalRemover()

def handler(job):
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "audio_data": "base64_encoded_audio",
            "filename": "audio.mp3",
            "method": "htdemucs_ft",
            "model_settings": {}
        }
    }
    """
    start_time = datetime.now()
    input_path = None
    output_path = None
    
    try:
        # Extract job input
        job_input = job.get("input", {})
        
        # Get parameters
        audio_data = job_input.get("audio_data")
        filename = job_input.get("filename", "audio.mp3")
        method = job_input.get("method", "htdemucs_ft")
        
        if not audio_data:
            return {"error": "No audio data provided"}
        
        logger.info(f"Processing serverless vocal removal: {filename}, method: {method}")
        
        # Decode audio file
        input_path = processor.decode_audio_file(audio_data, filename)
        
        # Process with GPU
        output_path = processor.remove_vocals_serverless(input_path, method)
        
        # Encode result
        result_data = processor.encode_audio_file(output_path)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = {
            "success": True,
            "processed_audio": result_data,
            "processing_time": processing_time,
            "method": method,
            "gpu_used": processor.device,
            "output_filename": f"{os.path.splitext(filename)[0]}_no_vocals.wav",
            "timestamp": datetime.now().isoformat(),
            "serverless": True
        }
        
        logger.info(f"Serverless vocal removal completed in {processing_time:.1f}s")
        
        # Cleanup
        processor.cleanup_files(input_path, os.path.dirname(output_path))
        
        return response
        
    except Exception as e:
        logger.error(f"Serverless handler error: {e}")
        
        # Cleanup on error
        if input_path:
            try:
                os.remove(input_path)
            except:
                pass
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": False,
            "error": str(e),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "serverless": True
        }

# Configure RunPod
runpod.serverless.start({"handler": handler})
