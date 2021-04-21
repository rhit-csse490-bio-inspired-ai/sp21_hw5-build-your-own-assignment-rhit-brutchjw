import os
import cv2
import numpy as np
import neat
import random
from labeler import Labeler
import possible_answers01 as pa
os.chdir('../..')
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'data\\Train\\01_shape_apple_banana')
training_images = []


def load_image(file):
    image = cv2.imread(file)
    image = cv2.resize(image, (16, 16))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = np.ndarray(shape=(1, 16, 16), dtype=int)
    image_array = np.asarray(gray)
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
    for genome_id, genome in genomes:
        genome.fitness = len(training_images)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for train in training_images:
            temp_input = train.image_array.flatten()
            output = net.activate(temp_input)
            genome.fitness -= (output[0] - pa.answers.index(train.answer)) ** 2


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
    pop.add_reporter(neat.Checkpointer(5))
    # Run for 50 generations
    winner = pop.run(eval_genomes, 50)
    # Display winner
    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for train in training_images:
        temp_input = train.image_array.flatten()
        output = winner_net.activate(temp_input)
        print("input {!r}, expected output {!r}, got {!r}".format(train.image_name, train.answer,
                                                                  pa.answers.__getitem__(round(output[0]))))


# if __name__ == '__main__':
load_and_label_training()
random.shuffle(training_images)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
run(config_path)
