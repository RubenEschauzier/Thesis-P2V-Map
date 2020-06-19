import sys
import datetime
import time
from pathlib import Path
import numpy as np
import pandas as pd
from termcolor import colored
from colorama import Fore, Style


def get_filepath(type_file, file_to_open):
    parent_directory = Path(type_file)
    file_path = parent_directory / file_to_open
    return file_path


def create_html(data, filename):
    df = pd.DataFrame(data)
    html = df.style.background_gradient(cmap='coolwarm').set_precision(2).render()
    text_file = open(filename, "w")
    text_file.write(html)
    text_file.close()


class INFO_LOGGING(object):

    def __init__(self):
        self.start_time = self.INFO_starting_simulation()
        pass

    def INFO_starting_simulation(self):
        time = self.get_current_time().strftime("%H:%M:%S")
        datetime_time = self.get_current_time()
        print("{} INFO: Starting simulation of product baskets".format(time), file = sys.stdout)
        return datetime_time

    def INFO_creating_consumers(self):
        time = self.get_current_time().strftime("%H:%M:%S")
        print("{} INFO: Creating consumer objects".format(time), file=sys.stdout)
        pass

    def INFO_simulation_epoch_done(self, week):
        time = self.get_current_time().strftime("%H:%M:%S")
        print("{} INFO: Starting simulation of week {}".format(time, week + 1), file=sys.stdout)

    def INFO_writing_to_file(self):
        time = self.get_current_time().strftime("%H:%M:%S")
        print("{} INFO: Writing simulated data to excel file".format(time), file=sys.stdout)

    def INFO_completed_simulation(self):
        time = self.get_current_time().strftime("%H:%M:%S")
        real_time = self.get_current_time()
        duration = real_time - self.start_time
        seconds = duration.seconds % 3600 % 60
        hours = duration.seconds//3600
        minutes = duration.seconds % 3600 // 60

        print("{} INFO: Completed simulation of product baskets, total time elapsed: {}"
              .format(time, "{}:{}:{}".format(hours, minutes, seconds), file=sys.stdout))

    @staticmethod
    def get_current_time():
        time = datetime.datetime.now()
        return time
