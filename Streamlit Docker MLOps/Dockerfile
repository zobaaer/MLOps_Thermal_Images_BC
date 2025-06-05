# Base image with Miniconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Make sure conda is initialized
SHELL ["conda", "run", "-n", "streamlit_app_env", "/bin/bash", "-c"]

# Copy your app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["conda", "run", "-n", "streamlit_app_env", "streamlit", "run", "app.py"]

