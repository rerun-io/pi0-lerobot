# At the very top of your "{your_package}.__init__" submodule:
from beartype.claw import beartype_this_package  # <-- boilerplate for victory

beartype_this_package()  # <-- yay! your team just won
