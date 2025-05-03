# Exam Model Project

This project is a Streamlit application designed to facilitate the grading of exams using a machine learning model. It allows users to input their name and ID, and generates the answer model expected by the grading system.

## Project Structure

- **src/**: Contains the source code for the Streamlit application.
  - **app.py**: The main entry point for the Streamlit application.
  - **utils/**: Contains utility functions for file operations.
    - **file_handlers.py**: Functions for reading JSON files.
  - **components/**: Contains reusable components for the Streamlit interface.
    - **exam_form.py**: Defines the exam form for user input.

- **data/**: Directory for storing JSON files related to exam questions and answers.

- **requirements.txt**: Lists the dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit application with:
   ```
   streamlit run src/app.py
   ```

## Usage Guidelines

- Upon running the application, users will be prompted to enter their name and ID.
- Users can then fill out the exam form based on the questions provided in the JSON files located in the `data/` directory.
- After submission, the application will generate the answer model that can be processed by the grading system.

## Contributing

Contributions to the project are welcome. Please submit a pull request or open an issue for any enhancements or bug fixes.