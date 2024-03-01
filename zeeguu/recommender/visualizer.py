import matplotlib.pyplot as plt
import numpy as np
from zeeguu.recommender.utils import  days_since_normalizer, get_expected_reading_time, resource_path, lower_bound_reading_speed, upper_bound_reading_speed
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class Visualizer:

    def get_diff_color(self, df, precise=False):
        if precise:
            return np.where(df['difficulty_feedback'] == 1, 'yellow', np.where(df['difficulty_feedback'] == 3, 'blue', 'black'))
        else:
            return "black"

    def plot_urs_with_duration_and_word_count(self, df, have_read_sessions, file_name, simple=False):
        plt.clf()
        if len(df) == 0:
            print("No data to plot")
            return
        
        x_min, x_max = 0, 2000
        y_min, y_max = 0, 2000

        plt.xlabel('Word count')
        plt.ylabel('Duration')

        expected_read_color = np.where(df['liked'] == 1, 'green', 
                                    np.where(df['difficulty_feedback'] != 0, self.get_diff_color(df, True),
                                        np.where(df['expected_read'] == 1, 'blue', 'red')))
        plt.scatter(df['word_count'], df['session_duration'], alpha=[days_since_normalizer(d) for d in df['days_since']], color=expected_read_color)

        x_values = df['word_count']
        y_values_line = [get_expected_reading_time(x, lower_bound_reading_speed) for x in x_values]
        plt.plot(x_values, y_values_line, color='red', label='y = ')

        x_values = df['word_count']
        y_values_line = [get_expected_reading_time(x, upper_bound_reading_speed) for x in x_values]
        plt.plot(x_values, y_values_line, color='red', label='y = ')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        plt.rc('axes', axisbelow=True)

        if have_read_sessions > 0:
            have_read_ratio = have_read_sessions / len(df) * 100
            have_not_read_ratio = 100 - have_read_ratio
            plt.text(0, 1.1, f"Have read: {have_read_ratio:.2f}%", transform=plt.gca().transAxes)
            plt.text(0, 1.05, f"Have not read: {have_not_read_ratio:.2f}%", transform=plt.gca().transAxes)
            plt.text(0, 1.01, f"Green = liked, yellow = easy, blue = Ok, black = Difficult", transform=plt.gca().transAxes)
        if simple:
            plt.text(0, 1.01, f"Green = liked, yellow = easy, blue = Ok, black = Difficult", transform=plt.gca().transAxes)

        #Change to '.svg' and format to 'svg' for svg.
        plt.savefig(resource_path + file_name + '.png', format='png', dpi=900)
        print("Saving file: " + file_name + ".png")
        plt.show()
    
    def visualize_tensor(self, tensor, file_name='tensor'):
        # This method save a .png image that shows the value of each user-article pair, by using color to represent the value.
        print("Visualizing tensor")

        with tf.Session() as sess:
            indices = sess.run(tensor.indices)
            values = sess.run(tensor.values)

            plt.xlabel('User id')
            plt.ylabel('Article id')

            # Plot values from Tensor
            plt.scatter(indices[:, 0], indices[:, 1], c=values)
            plt.title('Sparse Tensor')

            # Plot Density
            '''density = len(values) / (dense_shape[0] * dense_shape[1])
            axs[2].text(0.5, 0.5, f'Density: {density:.2f}', fontsize=12, ha='center')
            axs[2].axis('off')
            axs[2].set_title('Density')'''

            plt.savefig(resource_path + file_name + '.png', format='png', dpi=900)
            print("Saving file: " + file_name + ".png")
            plt.show()
