Environment Methods
===============================================
baseNetwork()
    BaseNetwork Definitition: The default network that all the AI share when they are created
    Setting the basenetwork mid-evolution will reset them all to the new baseNetwork
    - Empty Arg: Returns the set baseNetwork
    - Arg (network): Replaces the set baseNetwork with a new baseNetwork
    Errors:
        "Invalid Input: bad .baseNetwork() argument"
            - The inputed network doesn't match the format
        "Invalid Input: .baseNetwork() argument must be an network list or empty"
            - The inputed network isn't a network. Parent object must be a list

best()
    Returns the best ai in the generation
    - If a generation is not yet ran, it will return the best of the previous generation
    - Arg (str): best("excludePreviousBest") will return the best of the current generation, ignoring the previous generation's best
    Errors:
        Setup Error: .best() only works after the environment inputs and outputs are defined
            - The environment creates the ai after the inputs and outputs are defined
            - Because of this, the best() function only works after the inputs and outputs are defined
            - Define the inputs and outputs before running to fix this error

genSize()
    GenSize Definition: How large each generation size is (how many ai run in a single generation)
    - Empty Arg: Returns the current generation size
    - Argument (int): Sets the generation size to the argument. Creates and deletes ai if needed
    Errors:
        "Invalid Input: .genSize() argument must be an int or empty"

getNeuronIDs()
    - Used inside the module. Not for user use

hiddenAF()
    Hidden AF Definition: The activation function that will run on the inside layers of the neural network
    Activation functions add up all the inputs to a neurons and creates an output from that
    Look up the standard af's under the Activation Functions header
    - Empty arg: Returns the currently set hidden activation function
    - Argument (str): Sets a new hidden activation function
    Errors:
        "Input Error: Activation function from .hiddenAF() not found in af list"
        "Invalid Input: .hiddenAF() argument must be a string or empty"

outputAF()
    Output AF Definition: The activation function that will run on the output layers of the neural network
    Activation functions add up all the inputs to a neurons and creates an output from that
    Look up the standard af's under the Activation Functions header
    - Empty arg: Returns the currently set output activation function
    - Argument (str): Sets a new output activation function
    Errors:
        "Input Error: Activation function from .outputAF() not found in af list"
        "Invalid Input: .outputAF() argument must be a string or empty"

#customAF()
#    Adds in a custom activation function that can be used by the ai like the others
#    Note: Users need to include any imports from other modules/libaries inside of the function
#    - Empty Arg: Returns the current custom activation functions (even if not used)
#    - Argument 1 (string): Name that will be used to refer to the function
#    - Argument 2 (function): Function that will be run when the activation function is called
#    Ex] Environment.customAF("leakyReLU", leakyrelu)
#    Argument 1 (string): Name of the activation function
#    Argument 2 (function): The activation function
#    Errors:
#        "Invalid Input: First argument (function name) of .customAF() must be a string"

inputs()
    Inputs Definition: The inputs define the fields that the ai will accept when ran. 
    For example: If you do inputs(["brightness"]), your run function should look like this:
        Brain.run({"brightness": someNumber})
    Any extra inputs put into the run function will simply be ignored
    - Empty Arg: Returns the current inputs
    - Argument (list): Sets the new inputs as the list
    Errors:
        "Invalid Input: .inputs() argument must be a list or empty"
    
    ######## inputs are often really large numbers, will this cause an issue?

outputs()
    Output Definition: Outputs define the fields in the resulting dictionary after a brain is ran
    For example: If you do outputs(["lightOff", "lightOn"]) and run the brain, the output should look like this:
        Brain.run({inputsHere})    ----Returns--->     {"lightOff": someNumber, "lightOn": someNumber}
    - Empty Arg: Returns the current output list
    - Argument (list): set the new outputs as the list
    Errors:
        "Invalid Input: .outputs() argument must be a list or empty"

mutate()
    Mutate Definition: Changes the (referenced) inputed network in a random way 
    - Argument 1 (network): The network you want to mutate
    - Argument 2 (optional bool): True if you want multiple mutations, False or empty if you only want a single mutation
    Note: A brain's network cannot be mutated directly. 
          It must be copied with the brain.network() function. 
          Mutate that and create a new brain with the mutated network.
    Errors:
        "Invalid Input: .mutate() first argument must be a list"
        "Invalid Input: .mutate() second argument must be a bool or empty"
        "Input Error: .mutate() multiple mutations (second argument) must True in order to have a mutationChance (third argument)"
        "Input Error: .mutate() mutation chance (third argument) must a float between 0 and 1 in order to have multiple mutations (second argument)"

mutateOnce()
    - Used in the module. Not for user use

mutationChance()
    Mutation Chance Definition: The likelyhood a mutation will occur. 
    When doing multiple mutations, the mutations will stop when a mutation is not chosen
    So if you have it as 0.75 (aka 75%), there will be a 25% on each mutation iteration that it'll stop
    - Empty Arg: Returns the current mutation chance
    - Argument (float between 0 and 1): Sets the mutation chance to the argument
    Errors:
        "Invalid Input: .mutationChance() argument must be a floating point between 0 and 1.0"

mutationType()
    Single: Only one mutation will happen per brain (when creating the next gen)
    Multiple: Mutations will happen until no mutation is chosen (when creating the next gen)
    -
    Errors:
        "Input Error: .mutationType() argument must be either 'single' or 'multiple' (string)"
        "Invalid Input: .mutationType() argument must be either 'single' or 'multiple'"

size()
    Mainly for module use
    Used to determine how large a network is based on a point system
    Idealy, the smaller the network the better
    Point System | Weight: +5, Neuron: +15, Layer: +20
    Argument (network): Returns how large a network is
    Errors:
        "Invalid Input: .size() argument must be an network list or empty"

status()
    Returns a string displaying all the setting and set arguments
    Errors (returned as a string):
        "Critical Warning: deepCopy() does not exist"
            The deepCopy() function failed to import
        "Warning: Inputs are undefined"
        "Warning: Outputs are undefined"
        "Module Error: Defined generation size doesn't match generation list length"

readyAI()
    ##### Sooooo janky

nextGen()
    creates the next generation
    can expand on this more in the actual documentation
    ## Added if its the first generation, then nothing happens


.genCount
    Returns the generation iteration you are currently on

.gen
    Returns the list containing all the brain ai
    This is used heavily
    See examples of how this is used
    #### Check if this is the same size as the genSize



Brain Methods
===============================================
network()
    Returns a deep copy of the expanded network that the brain is using
    The network will have the neuron classes expanded into their dictionary forms

run()
    Returns the outputs as a dictionary with outputName:value pairs
    Argument (dict): Dictionary with inputName:value pairs
    Errors:
        "Invalid Input: .run() argument must be a dictionary with inputName:value pairs"
    #### Need to add a check for if an input can be inputed or not

#customAF()
#    Adds in a custom activation function that can be used by the ai like the others
#    Note: Users need to include any imports from other modules/libaries inside of the function
#    Ex] Environment.customAF("leakyReLU", leakyrelu)
#    Argument 1 (string): Name of the activation function
#    Argument 2 (function): The activation function
#    Errors:
#        "Input Error: Both arguments (name and function) of .customAF must be defined"
#        "Invalid Input: First argument (function name) of .customAF() must be a string"
#        "Input Error: Error when testing custom af in .customAF. Make sure there is an argument for a float"
#        "Input Error: New af from .customAF must output an int or float"

addFunction



-- will the mutation happen only after they are set up, actually, that might be ok
-- comments should be no longer than 72 char, go onto next line if neeeded
-- can put triple quotes at the start of a script/class/function/anything to make it the __doc__ and in help(arg)
-- https://realpython.com/documenting-python-code/
-- 72 chr does not include spaces








Instances where setupAI() should be called
==========================================
- change in inputs
- change in outputs (?)
- basenetwork is changed
- new gensize

SetupAI():
-----------
- only runs if outputs and inputs are defined
- creates new ai
* doesn't update
*Inputs and Outputs update

- find differences from genSize appended (+others) and mutate







