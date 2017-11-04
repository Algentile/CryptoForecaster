from parsers.data_parser import get_user_input
from models.recurrent_model import CryptoLSTM

def main():

	file_dict = get_user_input()
	train_input, train_output = file_dict['train']
	test_input, test_output = file_dict['test']

	lstm_model = CryptoLSTM(train_input, train_output, test_input, test_output)

	lstm_model.construct_neural_network()

if __name__ == '__main__':
	main()