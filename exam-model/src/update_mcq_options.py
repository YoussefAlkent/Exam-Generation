import json
import os

def update_mcq_options():
    """
    Script to update the MCQ questions in the exam data with options.
    """
    data_dir = './data'
    
    # Sample options for GAN-related MCQ questions
    mcq_options = {
        "In GANs, what problem does W-Loss primarily address?": [
            "Mode collapse",
            "Vanishing gradients",
            "Exploding gradients",
            "All of the above"
        ],
        "What type of values does the discriminator output in a GAN using BCE Loss?": [
            "Probabilities between 0 and 1",
            "Real numbers between -∞ and +∞",
            "Integer values",
            "Binary values (0 or 1)"
        ],
        "What is a key characteristic of activation functions used in neural networks, including GANs?": [
            "Non-linearity",
            "Symmetry",
            "Differentiability",
            "All of the above"
        ],
        "What is the Earth Mover's Distance (EMD) used for in the context of GANs?": [
            "Measuring distribution similarity",
            "Optimizing generator weights",
            "Calculating gradient updates",
            "Normalizing input data"
        ],
        "During testing with Batch Normalization, which statistics are used?": [
            "Running mean and variance from training",
            "Batch statistics from the test set",
            "Fixed values of 0 and 1",
            "Randomly sampled values"
        ]
    }
    
    # Loop through all JSON files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            
            # Read the existing exam data
            with open(file_path, 'r') as f:
                exam_data = json.load(f)
            
            # Update MCQ questions with options
            updated = False
            for question in exam_data.get("questions", []):
                if question.get("type") == "mcq" and question.get("question") in mcq_options:
                    # Add options if they don't exist or are empty
                    if not question.get("options"):
                        question["options"] = mcq_options[question["question"]]
                        updated = True
            
            # Save the updated exam data if changes were made
            if updated:
                with open(file_path, 'w') as f:
                    json.dump(exam_data, f, indent=2)
                print(f"Updated options in {filename}")
            else:
                print(f"No updates needed for {filename}")

if __name__ == "__main__":
    update_mcq_options()
    print("MCQ options update complete!")