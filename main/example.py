import sys
from parsers.data_parser import get_config_map, parse_csv
from models.recurrent_model import CryptoLSTM


def main():
    config_file = sys.argv[1]
    settings_map = get_config_map(config_file)

    (train_input, train_output) = parse_csv(
        settings_map['train_file'], column_number=3)
    (test_input, test_output) = parse_csv(
        settings_map['test_file'], column_number=3)

    lstm_model = CryptoLSTM(train_input, train_output, test_input, test_output)

    lstm_model.construct_neural_network()

if __name__ == '__main__':
    main()
