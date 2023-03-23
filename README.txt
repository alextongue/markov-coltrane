Second Order Markov Model Chain

[INSTRUCTIONS]
1. Open a terminal to this directory and start Julia.
2. In the Julia REPL, type ']' and enter "activate ." to activate the environment. This uses a few packages that are included in the "general registries".
3. Press backspace to return to the REPL and enter: include("markovtrane.jl")
4. A file named "out.mid" will be written to this directory. Figures should also come up, but I'm not sure if your shell runs the same as the my computer.

[NOTE 1] If you'd like to play with some parameters, they are included in the first code block of the file. I have only tested this up to second order (predicting a note based on previous two notes), but you are welcome to change the first two notes that prime this framework. It takes inMIDI note values.

[NOTE 2] There is an option to construct a separate train/inference Markov model for velocities. I find that it's a little more sensitive to being primed right, but there are instructions to set that up in the beginning of the script as well.