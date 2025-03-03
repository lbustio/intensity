import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class DataVisualizer:
    """
    A class to represent and plot different types of charts using matplotlib and seaborn.
    
    Attributes:
        data (pandas.DataFrame): The dataset to visualize.
        logger (logging.Logger): Logger instance to log messages during the plotting.
        
    Methods:
        scatter_plot(x, y, title="Scatter Plot", xlabel=None, ylabel=None, save_to=None, legend=False):
            Creates a scatter plot for the given data columns.
        line_plot(x, y, title="Line Plot", xlabel=None, ylabel=None, save_to=None, legend=False):
            Creates a line plot for the given data columns.
        bar_plot(x, y, title="Bar Plot", xlabel=None, ylabel=None, save_to=None, legend=False):
            Creates a bar plot for the given data columns.
        histogram(column, bins=10, title="Histogram", xlabel=None, ylabel=None, save_to=None):
            Creates a histogram for the given column of data.
        box_plot(column, title="Box Plot", xlabel=None, ylabel=None, save_to=None):
            Creates a box plot for the given column of data.
        heatmap(correlation_matrix, title="Heatmap", save_to=None):
            Creates a heatmap for the given correlation matrix.
        pie_chart(column, title="Pie Chart", save_to=None):
            Creates a pie chart for the given categorical column.
        scatter_plot_3d(x, y, z, title="3D Scatter Plot", xlabel=None, ylabel=None, zlabel=None, save_to=None):
            Creates a 3D scatter plot for the given data columns.
        surface_plot(x, y, z, title="Surface Plot", xlabel=None, ylabel=None, zlabel=None, save_to=None):
            Creates a surface plot for the given data columns.
        radar_chart(categories, values, title="Radar Chart", save_to=None):
            Creates a radar chart for the given categories and values.
    """

    def __init__(self, data, logger=None):
        """
        Constructor to initialize the DataVisualizer class.
        
        :param data: A pandas DataFrame containing the data to visualize.
        :param logger: A logger instance to log execution messages.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        
        self.data = data
        self.logger = logger  # Store the logger

        if self.logger:
            self.logger.info("DataVisualizer instance created successfully.")

    def scatter_plot(self, x, y, title="Scatter Plot", xlabel=None, ylabel=None, save_to=None, legend=False):
        """
        Creates a scatter plot for the given data columns.
        
        :param x: The column name for the x-axis.
        :param y: The column name for the y-axis.
        :param title: The title of the plot.
        :param xlabel: The label for the x-axis.
        :param ylabel: The label for the y-axis.
        :param save_to: Filepath to save the plot, if provided.
        :param legend: Whether to show the legend (default: False).
        """
        if self.logger:
            self.logger.info(f"Creating scatter plot for {x} vs {y}.")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.data, x=x, y=y)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x)
        plt.ylabel(ylabel if ylabel else y)
        
        if legend:
            plt.legend(title="Legend", loc="best")
        
        if save_to:
            plt.savefig(save_to)
            if self.logger:
                self.logger.info(f"Scatter plot saved to {save_to}.")
        
        plt.show()
        if self.logger:
            self.logger.info("Scatter plot created and displayed.")

    def scatter_plot_3d(self, x, y, z, title="3D Scatter Plot", xlabel=None, ylabel=None, zlabel=None, save_to=None):
        """
        Creates a 3D scatter plot for the given data columns.
        
        :param x: The column name for the x-axis.
        :param y: The column name for the y-axis.
        :param z: The column name for the z-axis.
        :param title: The title of the plot.
        :param xlabel: The label for the x-axis.
        :param ylabel: The label for the y-axis.
        :param zlabel: The label for the z-axis.
        :param save_to: Filepath to save the plot, if provided.
        """
        if self.logger:
            self.logger.info(f"Creating 3D scatter plot for {x}, {y}, {z}.")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data[x], self.data[y], self.data[z])
        
        ax.set_title(title)
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else y)
        ax.set_zlabel(zlabel if zlabel else z)
        
        if save_to:
            plt.savefig(save_to)
            if self.logger:
                self.logger.info(f"3D scatter plot saved to {save_to}.")
        
        plt.show()
        if self.logger:
            self.logger.info("3D scatter plot created and displayed.")

    def surface_plot(self, x, y, z, title="Surface Plot", xlabel=None, ylabel=None, zlabel=None, save_to=None):
        """
        Creates a surface plot for the given data columns.
        
        :param x: The column name for the x-axis.
        :param y: The column name for the y-axis.
        :param z: The column name for the z-axis.
        :param title: The title of the plot.
        :param xlabel: The label for the x-axis.
        :param ylabel: The label for the y-axis.
        :param zlabel: The label for the z-axis.
        :param save_to: Filepath to save the plot, if provided.
        """
        if self.logger:
            self.logger.info(f"Creating surface plot for {x}, {y}, {z}.")
        
        # Create grid for the surface plot
        x_vals = np.linspace(self.data[x].min(), self.data[x].max(), 100)
        y_vals = np.linspace(self.data[y].min(), self.data[y].max(), 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.interp(X, self.data[x], self.data[z])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_title(title)
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else y)
        ax.set_zlabel(zlabel if zlabel else z)
        
        if save_to:
            plt.savefig(save_to)
            if self.logger:
                self.logger.info(f"Surface plot saved to {save_to}.")
        
        plt.show()
        if self.logger:
            self.logger.info("Surface plot created and displayed.")

    def radar_chart(self, categories, values, title="Radar Chart", save_to=None):
        """
        Creates a radar chart for the given categories and values.
        
        :param categories: The categories to display on the radar chart.
        :param values: The values associated with each category.
        :param title: The title of the plot.
        :param save_to: Filepath to save the plot, if provided.
        """
        if self.logger:
            self.logger.info("Creating radar chart.")
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Make the radar chart circular
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        
        plt.title(title)
        
        if save_to:
            plt.savefig(save_to)
            if self.logger:
                self.logger.info(f"Radar chart saved to {save_to}.")
        
        plt.show()
        if self.logger:
            self.logger.info("Radar chart created and displayed.")
