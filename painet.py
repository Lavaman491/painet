"""
Nearly every concept is from Code Bullet on youtube
---------------------------------------------------
Code Bullet: https://www.youtube.com/@CodeBullet
How AI Learn (Genetic Algorithm): https://www.youtube.com/watch?v=VnwjxityDLQ
How AI Learn (Coded Example): https://www.youtube.com/watch?v=BOZfhUcNiqk
What Are Neural Networks: https://www.youtube.com/watch?v=JeVDjExBf7Y


Note: Skulpt (trinket.io) doesn't like a class having a class tag and a method 
sharing the same name. This is why Environment.Input is capitalized

Possible Implementation: Could have most publically used tags capitalized and
give a method for the user to edit them. This could uniform the tags a bit more


Just some text or something up here
So, um... hi
Check out Lemon Demon's 'The Machine'
It's pretty cool
But don't feel obligated to or anything
Well... anyways...
I'm going to get back to this crazy ai thing
See you later~~~~~~



Notes:
--------
NeuroEvolution

Make a point to convey that the score should be of two parts
    A - A really high but static number that determines if it made it
  B - A smaller variable score to how close or what you want it to improve at

"""

#----------------------------------------------------------------------------------

class Environment():
  
    # Set up functions
    #=================
    def __init__(self):
        # For nextGen()
        from random import uniform
        self.uniform = uniform
        # For mutateOnce()
        from random import randint, random, randrange, choice
        self.randint = randint
        self.random = random
        self.randrange = randrange
        self.choice = choice
        # Add deepCopy() function
        try:
            deepCopy()
            self.deepCopy = deepCopy
        except:
            from painet import deepCopy
            self.deepCopy = deepCopy
        # Various Variables
        self.gen = []
        self.GenSize = 10
        self.Inputs = []
        self.Outputs = []
        self.HiddenAF = blankNetwork[0]['hiddenAF']
        self.OutputAF = blankNetwork[0]['outputAF']
        self.BaseNetwork = deepCopy(blankNetwork)
        self.MutationType = "single"
        self.MutationChance = 0.7
        self.genCount = 0
    
    
    def inputs(self, inputs=None):
        # Output 
        if inputs == None:
            return self.Inputs
        # Change 
        elif type(inputs) == list:
            self.Inputs = self.deepCopy(inputs)
            self.BaseNetwork[0]['inputs'] = self.Inputs
            for ai in self.gen:
                ai.Network[0]['inputs'] = self.Inputs
                self.readyAI()
        # Error
        else:
            raise "Invalid Input: .inputs() argument must be a list or empty"
    
    
    def outputs(self, outputs=None): 
        if outputs == None:
            return self.Outputs
        elif type(outputs) == list:
            for output in outputs:
                self.BaseNetwork[-2].append({"id": output, "weights": {"B": 0}})
            self.Outputs = self.deepCopy(outputs)
            self.BaseNetwork[-1] = self.Outputs
            for ai in self.gen:
                ai.Network[-1] = self.Outputs
                # Add in any added outputs and skip any pre-existing output neurons
                for output in outputs:
                    for neuron in ai.network()[-2]:
                        if neuron['id'] == output:
                            continue
                        else:
                            ai.Network[-2].append(
                                ai.Neuron({"id": output, "weights": {"B": 0}}, 
                                ai.Network[0]['outputAF']
                            ))
                # Get rid of any removed outputs
                for output in ai.Network[-2]:
                    if not output.ID in outputs:
                        del output
            self.readyAI()
        else:
            raise "Invalid Input: .outputs() argument must be a list or empty"
    
    
    def readyAI(self):
        if (self.Inputs) and (self.Outputs):   # Empty lists return False
            for i in range(self.GenSize-len(self.gen)):
                network = self.deepCopy(self.BaseNetwork)
                self.mutate(network, True)
                self.gen.append(Brain(network))
            
            
    
    # Redefine Value Functions
    #=========================
    def genSize(self, size=None):
        if size == None:
            return self.GenSize
        elif type(size) == int:
            if size == len(self.gen):
                pass
            elif size < len(self.gen):
                while not size == len(self.gen):
                    self.gen.append(Brain(self.BaseNetwork))
                self.readyAI()
            elif size > len(self.gen):
                while not size == len(self.gen):
                    self.gen.pop(-1)
        else:
            raise "Invalid Input: .genSize() argument must be an int or empty"

    
    def baseNetwork(self, network=None):
        if network == None:
            return self.deepCopy(self.BaseNetwork)
        elif type(network) == list:
            self.BaseNetwork = self.deepCopy(network)
            self.gen = []
            self.Inputs = network[0]['inputs']
            self.Outputs = network[1]
            self.HiddenAF = network[0]['hiddenAF']
            self.OutputAF = network[0]['outputAF']
            self.readyAI()
        else:
            raise "Invalid Input: .baseNetwork() argument must be an network list or empty"
    
    
    def hiddenAF(self, af=None):
        if af == None:
            return self.BaseNetwork[0]['hiddenAF']
        elif type(af) == str:
            self.HiddenAF = af
            self.BaseNetwork[0]['hiddenAF'] = af
            for ai in self.gen:
                ai.Network[0]['outputAF'] = af
                for layer in ai[1:-2]:
                    for neuron in layer:
                        neuron.AF = af
        else:
            raise "Invalid Input: .hiddenAF() argument must be a string or empty"
        
      
    def outputAF(self, af=None):  
        if af == None:
            return self.BaseNetwork[0]['outputAF']
        elif type(af) == str:
            self.OutputAF = af
            self.BaseNetwork[0]['outputAF'] = af
            for ai in self.gen:
                ai.Network[0]['outputAF'] = af
                for neuron in ai[-2]:
                    neuron.AF = af
        else:
            raise "Invalid Input: .outputAF() argument must be a string or empty"
    
    
    def mutationType(self, arg=None):
        if arg == None:
            return self.MutationType
        elif type(arg) != str:
            raise "Invalid Input: .mutationType() argument must be either 'single' or 'multiple'"
        elif arg.lower() == "single":
            self.MutationType = "single"
        elif arg.lower() == "multiple":
            self.MutationType = "multiple"
        else:
            raise "Invalid Input: .mutationType() argument must be either 'single' or 'multiple'"
            
      
    def mutationChance(self, arg=None):
        if arg == None:
            return self.MutationChance
        elif type(arg) == float:
            if arg < 0 or arg > 1:
                raise "Invalid Input: .mutationChance() argument must be a floating point between 0 and 1.0"
            else:
                self.MutationChance = arg
        else:
            raise "Invalid Input: .mutationChance() argument must be a floating point between 0 and 1.0"
    
      
      
    # Return Output Functions
    #========================
    def status(self):
        if len(self.BaseNetwork[0]['inputs']) == 0:
            return "Warning: Inputs are undefined"
        elif len(self.BaseNetwork[-1]) == 0:
            return "Warning: Outputs are undefined"
        elif self.GenSize != len(self.gen):
            return "Module Error: Defined generation size doesn't match generation list length"
        else:
            return ("""Inputs are Defined;
                      Outputs are Defined;
                      Generation Size: %s;
                      Current Generation Iteration: %s;
                      Hidden Layer Activation Function Type: %s;
                      Output Layer Activation Function Type: %s;
                      Mutation Type: %s;
                      Mutation Chance: %s%%;
                      --Use Environment.baseNetwork() to check the base network--
                    """.replace("  ", "").strip() %(
                        str(self.GenSize), 
                        str(self.genCount), 
                        self.BaseNetwork[0]['hiddenAF'], 
                        self.BaseNetwork[0]['outputAF'],
                        self.MutationType,
                        int(self.MutationChance*100))
                  )
                  
                 
    def best(self, arg=None):
        if self.gen[0] == None:
            return self.gen[0]
        best = self.gen[0]
        for ai in self.gen[1:]:
            if arg == "excludePreviousBest" and ai.best:
                continue
            if ai.fitness == None:
                continue
            if ai.fitness > best.fitness:
                best = ai
            if ai.fitness == best.fitness:
                if self.size(ai.network()) < self.size(best.network()):
                    best = ai
        return best
        
    
    def size(self, network):
        size = 0
        for layer in network[1:-1]:
            for neuron in layer:
                for weight in neuron['weights']:
                    size += 5
                size += 15
            size += 20
        return size
    
    
    # Next Generation
    #=================
    def nextGen(self):
        # Sets up best and cutoff
        best = self.best()
        bestNet = self.deepCopy(best.network())
        if self.gen[0] is best:
            cutoff = self.best("excludePreviousBest").fitness
        else:
            cutoff = self.gen[0].fitness
      
        # Set up dictionary with fitness: network
        fitnesses = {}
        totalFitness = 0
        for ai in self.gen:
            if ai.fitness == None:
                continue
            else:
                # If previous best is still #1: only previous best and current 2nd best
                # If previous best is beathen: filters for better than previous best
                if ai.fitness < cutoff:
                    continue
                # Makes sure there aren't any duplicate keys
                interval = 0
                while ai.fitness in fitnesses:
                    interval += 1
                    if self.size(ai.network()) >= self.size(fitnesses[ai.fitness]):
                        ai.fitness -= 0.1/interval
                    else:
                        ai.fitness += 0.1/interval
                # Adds to fitnesses dictionary
                fitnesses[ai.fitness] = self.deepCopy(ai.network())
                totalFitness += ai.fitness
        
        # Create first AI of previous's best 
        # NOTE: The first ai is untouched to preserve learned networks
        self.gen = []
        ai = Brain(bestNet, True)    # True arg marks it as the previous best
        self.gen.append(ai)
        
        # Chooses Parent, Creates AI, and Mutates Network
        while len(self.gen) < self.GenSize:
            num = self.uniform(0, totalFitness)
            count = 0
            for fitness in fitnesses:
                count += fitness
                if num < count:
                    network = self.deepCopy(fitnesses[fitness])
                    if self.MutationType == "single":
                        self.mutate(network)
                    elif self.MutationType == "multiple":
                        self.mutate(network, True)
                    self.gen.append(Brain(network))
                    break



    # Mutate
    #=======
    def mutate(self, network, multipleMutations=False):
        # "And, what do we need to do with them? Say it with me. 3, 2, 1"
        # "We mutate those babies!  ...sorry that's a terrible thing to say" 
        # "But it doesn't mean it's not true." - Code Bullet
        if multipleMutations:
            msg = self.mutate(network)
            while msg != "QUIT":
                msg = self.mutateOnce(network, self.MutationChance)
        else:
            self.mutateOnce(network)
  
  
    def mutateOnce(self, network, stoppingChance=None):
        # This function is run by pure magic and stupidity. I'm not sure how I wrote it...
        message = "!NULL"
        while message[0] == "!":
            message = "!NULL"
            if type(stoppingChance) == float:
                 # Finds the random max number to create the mutation chance
                stoppingChance = 7/stoppingChance  
                if stoppingChance > 7:
                    mutationNum = self.randint(0,stoppingChance)
                else:
                    mutationNum = self.randint(0,7)
            else:
                mutationNum = self.randint(0,7)
            
            # ADD NEURON
            if mutationNum == 0:
                if self.randint(0,2) == 0:
                    # Add New Layer
                    layer = self.randint(1,len(network)-2)
                    network.insert(layer, [])
                    message = " NEW LAYER[" + str(layer) + "]"
                else:
                    layer = self.randint(1,len(network)-3)
                    message = " LAYER[" + str(layer) + "]"
                # Pick Nueron ID
                neuronID = str(self.randint(0,99999))
                while neuronID in self.getNeuronIDs(network, "all"):
                    neuronID = str(self.random.randint(0,99999))
                # Pick Weights
                weights = {}
                weights[self.choice(self.getNeuronIDs(network, layer))] = round(self.random()*self.randrange(-1,2,2), 2)
                while True:
                    if self.randint(0,1) == 0:
                        weights[self.choice(self.getNeuronIDs(network, layer))] = round(self.random()*self.randrange(-1,2,2), 2)
                    else:
                        break
                # Add Neuron
                network[layer].append({'id': str(neuronID), 'weights': self.deepCopy(weights)})
                message = "++ADDED NEURON[" + neuronID + "] IN" + message
        
            # DELETE NEURON
            elif mutationNum == 1:
                layer = self.randint(1,len(network)-3)
                attempts = 0
                if network[layer] != []:
                    neuron = self.randint(0,len(network[layer])-1)
                    neuronID = network[layer][neuron]["id"]
                    message = "--DELETED NUERON[" + neuronID + "] IN LAYER[" + str(layer) + "]" 
                    network[layer].pop(neuron)
                    for layer in network[1:-1]:
                        for neuron in layer:
                            if neuronID in neuron["weights"]:
                                del neuron["weights"][neuronID]
                            if neuron["weights"] == {}:
                                neuronPos = layer.index(neuron)
                                layerPos = network.index(layer)
                                weight = self.choice(self.getNeuronIDs(network, layerPos))
                                while (weight in network[layerPos][neuronPos]["weights"]):
                                    weight = self.choice(self.getNeuronIDs(network, layerPos))
                                network[layerPos][neuronPos]["weights"][weight] = round(self.random()*self.randrange(-1,2,2), 2)
                elif len(network) == 4:
                    message = "!--DELETED NEURON FAIL[too few layers] IN LAYER[" + str(layer) + "]"
                else:
                    network.pop(layer)
                    message = "--DELETED LAYER[" + str(layer) + "]"
          
            # CHANGE WEIGHT
            elif mutationNum == 2 or mutationNum == 3:
                layer = self.randint(1,len(network)-2)
                if network[layer] == []:
                    message = "!*CHANGE WEIGHT FAIL[empty layer] IN L[" + str(layer) + "]"
                else:
                    neuron = self.randint(0,len(network[layer])-1)
                    weight = self.choice(list(network[layer][neuron]["weights"].keys()))
                    network[layer][neuron]["weights"][weight] = round(self.random()*self.randrange(-1,2,2), 2)
                    message = "*CHANGED WEIGHT[" + weight + "] IN NEURON[" + network[layer][neuron]["id"] + "] IN LAYER[" + str(layer) + "]"
          
            # ADD WEIGHT
            elif mutationNum == 4 or mutationNum == 5: 
                layer = self.randint(1,len(network)-2)
                if network[layer] == []:
                    message = "!+ADDED WEIGHT FAIL[empty layer] IN LAYER[" + str(layer) + "]"
                else:
                    attempt = 0
                    neuron = self.randint(0,len(network[layer])-1)
                    weight = self.choice(self.getNeuronIDs(network, layer))
                    while (weight in network[layer][neuron]["weights"]) and (attempt < 5):
                        weight = self.choice(self.getNeuronIDs(network, layer))
                        attempt += 1
                    if attempt >= 5:
                        message = "!+ADDED WEIGHT FAIL[too many attempts with duplicate weights] IN NEURON[" + network[layer][neuron]["id"] + "] IN LAYER[" + str(layer) + "]"
                    else:
                        network[layer][neuron]["weights"][weight] = round(self.random()*self.randrange(-1,2,2), 2)
                        message = "+ADDED WEIGHT[" + weight + "] IN NEURON[" + network[layer][neuron]["id"] + "] IN LAYER[" + str(layer) + "]"
            
            # DELETE WEIGHT
            elif mutationNum == 6 or mutationNum == 7:
                layer = self.randint(1,len(network)-2)
                if network[layer] == []:
                    message = "!-DELETED WEIGHT FAIL[empty layer] IN LAYER[" + str(layer) + "]"
                else:
                    neuron = self.randint(0,len(network[layer])-1)
                    if len(network[layer][neuron]["weights"]) == 1:
                        message = "!-DELETED WEIGHT FAIL[too few weights] IN NEURON[" + network[layer][neuron]["id"] +"]IN LAYER[" + str(layer) + "]"
                    else:
                        weight = self.choice(list(network[layer][neuron]["weights"].keys()))
                        del network[layer][neuron]["weights"][weight]
                        message = "-DELETED WEIGHT[" + weight + "] IN NEURON[" + network[layer][neuron]["id"] + "] IN LAYER[" + str(layer) + "]"
        
            # QUIT
            elif mutationNum > 7:    # Quit
                message = "QUIT"
    
        return message
        
        
    def getNeuronIDs(self, network, layers):
        neuronIDs = []
        for nInput in network[0]['inputs']:
            neuronIDs.append(nInput)
        if layers == "all":
            layers = len(network)-1
        elif layers == 1:
            return neuronIDs
        for layer in network[1:layers]:
            for neuron in layer:
                neuronIDs.append(neuron['id'])
        return neuronIDs



#----------------------------------------------------------------------------------

class Brain():
    """NOTE: This class can be imported alongside the deppCopy() function and used completely seperate from the rest of the module"""

  
    # Neuron Class
    #=============
    class Neuron():
      
        def __init__(self, data, af):
            # Sets up Neuron Data
            from math import exp
            self.exp = exp
            self.ID = data['id']
            self.af = af
            self.weights = data['weights']
            
        def run(self, memory):
            # Adds up inputs and squashing
            total = 0
            for weight in self.weights:
                total += memory[weight]*self.weights[weight]
                memory[self.ID] = self.af(total)
  
  
  
    # NeuralNetwork Setup
    #========================
    def __init__(self, network, best=False):
        #(math.exp for sigmoid function)
        from math import exp
        self.exp = exp
        
        # Add deepCopy() function
        try:
            deepCopy()
            self.deepCopy = deepCopy
        except:
            from painet import deepCopy
            self.deepCopy = deepCopy
        
        # Adds Working Tags
        self.hiddenAF = network[0]['hiddenAF']
        self.outputAF = network[0]['outputAF']
        self.Inputs = network[0]['inputs']
        self.Outputs = network[-1]          # Possible Outputs
        self.active = True
        self.fitness = None
        self.best = best
        
        # Sets Up Network
        self.Network = []
        self.Network.append(self.Inputs)
        self.AFs = {
            'step':self.step, 
            'linear':self.linear, 
            'relu':self.relu, 
            'sigmoid':self.sigmoid, 
            'mirroredStep':self.mirroredStep
        }
        
        # Creates Hidden Neurons
        for layer in network[1:-2]:
            self.Network.append([])
            for nData in layer:
                neuron = self.Neuron(nData, self.AFs[self.hiddenAF])
                self.Network[-1].append(neuron) 
        
        # Creates Output Neurons
        self.Network.append([])
        for nData in network[-2]:
            neuron = self.Neuron(nData, self.AFs[self.outputAF])
            self.Network[-1].append(neuron)
        
        # Appends Output List
        # And yes I know there's self.output and self.Output. It's 12:56 pm and my head hurts, don't yell at me
        self.Network.append(self.Outputs)
        self.outputs = {}       # Brain generated Outputs
  
  
  
    # Neural Network Output Functions
    #=================================
    def run(self, inputs):
        # Compute Neurons
        memory = {'B': 1}
        for inputVar in self.Network[0]:
            memory[inputVar] = inputs[inputVar]
        for layer in self.Network[1:-1]:
            for neuron in layer:
                neuron.run(memory)
        # Run Through Outputs
        self.outputs.clear()
        for neuron in self.Network[-1]:
            self.outputs[neuron] = memory[neuron]
        return self.outputs
    
    def network(self):
        network = []
        network.append({
            "inputs": self.deepCopy(self.Inputs),
            "hiddenAF": self.hiddenAF,
            "outputAF": self.outputAF,
        })
        for layer in self.Network[1:-1]:
            network.append([])
            for neuron in layer:
                network[-1].append(
                    {"id": neuron.ID, "weights": neuron.weights}
                )
        network.append(self.deepCopy(self.Network[-1]))
        return network
        
    
    
    # Activation Functions   
    #=====================
    def step(brain, total):
        if total > 0:
            return 1
        elif total <= 0:
            return 0
    def linear(brain, total):
        return total
    def relu(brain, total):
        if total > 1:
            return 1
        elif total < 0:
            return 0
        else:
            return total
    def mirroredStep(brain, total):
        if total > 0:
            return 1
        elif total < 0:
            return -1
        else:
            return 0
    def sigmoid(brain, total):
        return 1/(1+brain.exp(-total))



#----------------------------------------------------------------------------------

def deepCopy(IN=None):
    if IN == None:
        return True
    # Made my own deepCopy function because trinket.io doesn't have one, y'know, as one does
    if str(type(IN)) == "<class 'dict'>":
        OUT = {}
        for key in IN.keys():
            value = IN[key]
            copied_value = deepCopy(value)
            OUT[key] = copied_value
    elif str(type(IN)) == "<class 'list'>":
        OUT = []
        for item in IN:
            copied_item = deepCopy(item)
            OUT.append(copied_item)
    else: 
        OUT = IN
    return OUT


def printNet(net):
    ppNet = ""
    for layer in net[1:]:
        ppNet = ppNet + "[" + "\n"
        for neuron in layer:
            ppNet = ppNet + "    " + str(neuron) + "\n"
        ppNet = ppNet + "]," + "\n"
    return ppNet



#----------------------------------------------------------------------------------

blankNetwork = [
    {
        "inputs": [],
        "hiddenAF": "step",
        "outputAF": "mirroredStep",
    },
    [
        # Hidden Neurons (anything between the header and the possible outputs)
    ],
    [
        # Output Neurons 
    ],
    [
        # Possible Outputs
    ]
]


exampleNetwork = [
    {
        "inputs": ["food_x", "food_y", "blob_x", "blob_y"],
        "hiddenAF": "step",
        "outputAF": "mirroredStep",
    },
    [
        {"id": "aaa", "weights": {'food_x': 1}},
        {"id": "bbb", "weights": {'food_y': 1}},
        {"id": "ccc", "weights": {'food_x': -1}},
        {"id": "ddd", "weights": {'food_y': -1}},
    ],
    [
        {"id": "move_right", "weights": {'aaa': 1}},
        {"id": "move_left", "weights": {'ccc': 1}},
        {"id": "move_up", "weights": {'bbb': 1}},
        {"id": "move_down", "weights": {'ddd': 1}},
    ],
    [
        {"id": "move_horizontally", "weights": {'move_right': 1, "move_left": -1}},
        {"id": "move_vertically", "weights": {'move_up': 1, "move_down": -1}},
    ],
    [
        "move_vertically",
        "move_horizontally"
    ]
]