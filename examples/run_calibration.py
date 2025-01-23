# example_calibration_run.py
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from SmoothCache import DiffuserCalibrationHelper

def main():
    # Load pipeline
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # Initialize calibration helper
    calibration_helper = DiffuserCalibrationHelper(
        model=pipe.transformer,
        calibration_lookahead=3,
        calibration_threshold=0.15,
        log_file="smoothcache_schedules/diffuser_schedule.json"
    )

    # Enable calibration
    calibration_helper.enable()

    # Run pipeline normally
    words = ["Labrador retriever"]
    class_ids = pipe.get_label_ids(words)
    generator = torch.manual_seed(33)
    images = pipe(
        class_labels=class_ids,
        num_inference_steps=50,
        generator=generator
    ).images  # Normal pipeline call

    # Disable calibration and generate schedule
    calibration_helper.disable()

    print("Calibration complete. Schedule saved to smoothcache_schedules/diffuser_schedule.json")

    # breakpoint()
    images[0].save('generated_image_cached.png')    

if __name__ == "__main__":
    main()
