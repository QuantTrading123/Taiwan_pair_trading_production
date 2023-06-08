from decimal import Decimal

class Pair_Trading_Config:
    def __init__(self,r,t) -> None:
        self.REFERENCE_SYMBOL = r
        self.TARGET_SYMBOL = t
        self.FUTURE_DATE_SYMBOL = "R1"

        self.OPEN_THRESHOLD = 1.5

        self.STOP_LOSS_THRESHOLD = 10

        self.MA_WINDOW_SIZE = 6000
        self.TEST_SECOND = 60
            
        self.PRECISION_AMOUNT_REF = Decimal('0')
        
        self.PRECISION_PRICE_REF = Decimal('0.00')
        
        
        self.PRECISION_AMOUNT_TARGET = Decimal('0')
        
        self.PRECISION_PRICE_TARGET = Decimal('0.00')
        
        self.SLIPPAGE = 0.001

    