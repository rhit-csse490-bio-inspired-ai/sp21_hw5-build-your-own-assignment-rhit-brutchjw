import os
import cv2
import numpy as np
import neat
import random

import visualize
from labeler import Labeler
import possible_answers02 as pa

os.chdir('../..')
base_dir = os.getcwd()
viz_dir = os.path.join(base_dir, 'viz\\02_visualizations')
visualize.set_viz_dir(viz_dir)
train_dir = os.path.join(base_dir, 'data\\Train\\02_color_apple_orange')
training_images = []
average_accuracy = []
m = len(pa.answers) - 1


def load_image(file):
    image = cv2.imread(file)
    image = cv2.resize(image, (16, 16))
    data = np.ndarray(shape=(1, 16, 16, 3), dtype=int)
    image_array = np.asarray(image)
    data[0] = image_array
    return data[0]


def load_and_label_training():
    for directory in os.listdir(train_dir):
        current_dir = os.path.join(train_dir, directory)
        if os.path.isdir(current_dir):
            for filename in os.listdir(current_dir):
                f = os.path.join(current_dir, filename)
                if os.path.isfile(f):
                    for label in pa.answers:
                        if directory.__contains__(label):
                            training_images.append(Labeler(filename, load_image(f), label))


def eval_genomes(genomes, config):
    avg_accuracy = 0
    for genome_id, genome in genomes:
        genome.fitness = len(training_images)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        accuracy = 0
        for train in training_images:
            temp_input = train.image_array.flatten()
            output = net.activate(temp_input)
            genome.fitness -= (output[0] - (pa.answers.index(train.answer) / m)) ** 2
            if train.answer == pa.answers.__getitem__(round(output[0] * m)):
                accuracy += 1
        avg_accuracy += (accuracy / len(training_images))
    average_accuracy.append(avg_accuracy / len(genomes))


def run(config_file):
    # Load config file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Creates population
    pop = neat.Population(config)
    # Adds reporter
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    # // Uncomment below to add checkpoints
    # pop.add_reporter(neat.Checkpointer(5))
    # Run for 50 generations
    winner = pop.run(eval_genomes, 3)
    # Display winner
    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    accuracy = 0
    total = 0
    for train in training_images:
        temp_input = train.image_array.flatten()
        output = winner_net.activate(temp_input)
        print("input {!r}, expected output {!r}, got {!r}".format(train.image_name, train.answer,
                                                                  pa.answers.__getitem__(round(output[0] * m))))
        total += 1
        if train.answer == pa.answers.__getitem__(round(output[0] * m)):
            accuracy += 1
    print("Accuracy Percentage: ", (accuracy / total * 100))
    # visualize things
    visualize.plot_accuracy(average_accuracy, view=True)
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


load_and_label_training()
random.shuffle(training_images)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
run(config_path)
