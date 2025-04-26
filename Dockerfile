# Start from full Anaconda
FROM continuumio/anaconda3

# Set working directory
WORKDIR /app

# Copy environment.yml first
COPY environment.yml .

# Create environment
RUN conda env create -f environment.yml

# Activate environment in shell
SHELL ["conda", "run", "-n", "microswarm", "/bin/bash", "-c"]

# Copy the rest of your code
COPY . .
