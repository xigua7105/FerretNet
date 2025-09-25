from util.register import REGISTER
from .syn_tester import SynTester

TESTER = REGISTER("trainers")
TESTER.register_module(SynTester)
