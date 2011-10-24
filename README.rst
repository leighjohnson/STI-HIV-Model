Instructions for users of the STI-HIV model
###########################################

This note has been prepared for individuals who wish to verify the calculations performed using the STI-HIV model, as presented in the paper “The effect of changes in condom usage and antiretroviral treatment coverage on HIV incidence in South Africa: a model-based analysis” (Johnson, Hallett, Rehle and Dorrington). The note is not intended to provide a comprehensive guide on all aspects of the STI-HIV model – it only concerns those aspects of the model that are relevant to the paper. Queries regarding the model should be directed to Leigh Johnson (Leigh.Johnson at uct.ac.za). The model is very complex and is not user-friendly, and anyone wishing to publish estimates based on the model is therefore encouraged to contact the author to ensure that their interpretation of the model parameters and outputs is correct.

Platforms supported
-------------------

The code has been tested using the following compilers:

- Microsoft Visual C++ 2010 Express on Microsoft Windows
- MinGW's port of g++ on Microsoft Windows
- g++ on Linux

The program is written in standard ANSI/ISO C++ (C++03) and any C++ compiler should compile it. It is a console application without any operating system specific code. The compiler will generate several warnings. These can be ignored.

Installation
------------

If you don't have git, download the file latest.zip and unzip it.

If you do have git, simply fork the github directory: leighjohnson/STI-HIV-Model

Microsft Visual C++ 2010 Express
================================

Open the project file and build the program.

g++ under Linux and g++ under Windows using MinGW
=================================================

At a command prompt in the folder containing the project, type:

make 

Running the program
-------------------

In Visual C++, press CTRL-F5 or F5.

If you used MinGW, the executable program is called *stihiv.exe*.

In Linux the executable is called *stihiv*.

Further technical notes can be found in the file notes.rtf

Credits
-------

Leigh Johnson is the author of the STI HIV model coded in TSHISAv1.cpp and TSHISAv1.h.

This 3rd-party code is used by this project:

Agner Fog is the author of the mersenne twister code.

Barry Brown, James Lovato, and Kathy Russell are the authors of the statistics functions in StatFunctions.cpp and StatFunctions.h

.. math::

  x^2
  
Testing.
