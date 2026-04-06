# Natural Language Processing
## Run commands

Into your bash command line, copy and paste the following command:

~~~
git clone https://github.com/V1b1ngC0w/NLP_3.git
~~~
Then you must navigate to the newly clone NLP directory with the command:
~~~
cd NLP_3
~~~

Then to sync all of the library dependencies, copy and paste the following command:
~~~
uv sync
~~~
Now activate the virtual environment with this command:
~~~
source .venv/bin/activate
~~~
Now you are ready to run the program, which can be done with the command:
~~~
python main.py
~~~
When the program is running, it will ask if you want it to train the models from scratch. Since the transformers could not be uploaded to git, responding with 'y' or 'n' doesn't change the outcome, but an answer is needed for the program to start.
