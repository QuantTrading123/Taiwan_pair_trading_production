# Pair Trading Strategy with Shioaji

This program is a pair trading strategy that uses the Shioaji API to execute trades on the Taiwan Stock Exchange. It reads configuration data from a file named `config.py` and executes the trading strategy.

## Prerequisites

To use this program, you must have the following:

- Python 3.7 or higher installed on your system
- Shioaji API installed (`pip install shioaji`)
- A valid Shioaji account with API key and secret

## Installation

1. Clone or download the repository to your local machine.
2. Install the required packages by running `pip install -r requirements.txt` in the command line.
3. Create a file named `credentials.py` in the same directory as the main program with your Shioaji API key and secret in the following format:

```
api_key = 'your_api_key'
api_secret = 'your_api_secret'
```

4. Create a file named `config.py` in the same directory as the main program with the necessary configuration settings. Example:
```
class Pair_Trading_Config:
    REFERENCE_SYMBOL = "3037"
    TARGET_SYMBOL = "8069"
    FUTURE_DATE_SYMBOL = "202305"

    OPEN_THRESHOLD = 1.5

    STOP_LOSS_THRESHOLD = 5

    MA_WINDOW_SIZE = 100
        
    PRECISION_AMOUNT_REF = Decimal('0')
    
    PRECISION_PRICE_REF = Decimal('0.00')
    
    
    PRECISION_AMOUNT_TARGET = Decimal('0')
    
    PRECISION_PRICE_TARGET = Decimal('0.00')
    
    SLIPPAGE = 0.001
    TEST_SECOND = 6

    TEST_SYMBOL = '2317'
```
5. Run the program by running `python main.py` in the command line.

## Usage

The program will execute the pair trading strategy based on the configuration settings in `config.py`. You can modify the configuration settings as needed to customize the strategy.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact

If you have any questions or comments, please feel free to contact the project maintainer at [email address].




Regenerate response

