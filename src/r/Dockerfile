# Use an R environment
FROM rocker/r-ver:4.4.3

# Set the working directory
WORKDIR /app

# Copy R package installation script
COPY src/r/install_packages.R ./install_packages.R

# Install required R packages
RUN Rscript install_packages.R

# Copy the R model script
COPY src/r/main.R ./main.R

# Run the R script when the container starts
CMD ["Rscript", "main.R"]
