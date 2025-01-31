import sys 
from src.logger import logging
'''
The sys module in Python provides various functions and variables that are used to manipulate different parts of the python runtime environment. It allows operating on the interpreter as it provides access to the variables and functions that interact strongly with the interpreter
'''
def error_message_details(error, error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))

class CustomExpection(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_details=error_detail)
    
    def __str__(self):
        return self.error_message


