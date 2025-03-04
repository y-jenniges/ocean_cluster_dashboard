FROM jupyter/minimal-notebook

# Install dependencies
RUN pip install dash plotly numpy pandas gsw

# Copy the Python script into the container
COPY dashboard_ocean_cluster_visualisation.py /home/viz/

# Expose the Dash app port
EXPOSE 8050

# Command to run the Python Dash app
CMD ["python", "/home/viz/dashboard_ocean_cluster_visualisation.py"]
