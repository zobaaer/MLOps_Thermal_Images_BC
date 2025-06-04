# Use Mambaforge for fast conda installs
FROM condaforge/mambaforge:latest

# Set working directory
WORKDIR /app

# Copy environment file and app code
COPY environment.yml .
COPY . /app

# Create conda environment
RUN mamba env create -f environment.yml

# Activate environment and set PATH
SHELL ["conda", "run", "-n", "my_app_env", "/bin/bash", "-c"]

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
CMD ["conda", "run", "--no-capture-output", "-n", "my_app_env", "streamlit", "run", "app.py"]
