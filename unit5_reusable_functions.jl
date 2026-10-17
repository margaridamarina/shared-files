using Statistics
using Random
using Flux
using Flux: Losses

# Exact function copies from ML1_notebook3/scripts/reusable_functions.jl
# and ML1_notebook4/scripts/reusable_functions.jl (confusionMatrix methods).

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    @assert all([in(value, classes) for value in feature]) "Not all features belong to the provided classes"
    numClasses = length(classes)
    @assert numClasses > 1 "The number of classes must be greater than 1"

    if numClasses == 2
        oneHot = reshape(feature .== classes[1], :, 1)
    else
        oneHot = BitArray{2}(undef, length(feature), numClasses)
        for numClass in 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
    end
    return oneHot
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, 
                          normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1]
    maxValues = normalizationParameters[2]
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, 
                         normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(dataset), normalizationParameters)
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2)
    @assert numOutputs != 2 "Two-column matrices are not supported by convention"

    if numOutputs == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        booleanOutputs = falses(size(outputs))
        booleanOutputs[indicesMaxEachInstance] .= true
        @assert all(sum(booleanOutputs, dims=2) .== 1)
        return booleanOutputs
    end
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer]
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
        numInputsLayer = numNeurons
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert all(size(outputs) .== size(targets))
    if size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

function holdOut(N::Int, P::Real) 
    @assert N >= 0 && 0 <= P <= 1
    indices = randperm(N)
    ntest = round(Int, N * P)
    return indices[ntest+1:end], indices[1:ntest]
end

function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (zeros(Float32,0,0), falses(0,0)),
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (zeros(Float32,0,0), falses(0,0)),
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        maxEpochsVal::Int=20,
        showText::Bool=false)

    # Unpack the training, validation and test tuple: inputs are features (X), targets are desired labels (y).
    (inputs, targets) = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs, testTargets) = testDataset;

    @assert(size(inputs,1)==size(targets,1)); "Training inputs and targets must have matching sample counts"

    # Determine whether validation and test sets were provided (optional parameters)
    hasVal  = !isempty(validationInputs) && !isempty(validationTargets)
    hasTest = !isempty(testInputs) && !isempty(testTargets)

    # Define the ANN
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Loss function selection: binary cross-entropy (1 output) vs. categorical cross-entropy
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # This vectors are going to contain the losses and precission of each epoch
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Epoch 0 evaluation (before any weight updates)
    numEpoch = 0
    trainingLoss = loss(ann, inputs', targets')
    push!(trainingLosses, trainingLoss)

    if hasVal
        valLoss = loss(ann, validationInputs', validationTargets')
        push!(validationLosses, valLoss)
        bestLoss = valLoss
        bestANN  = deepcopy(ann)
    end

    if hasTest
        testLoss = loss(ann, testInputs', testTargets')
        push!(testLosses, testLoss)
    end

    showText && println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    # Tracking variables for Early Stopping
    epochsWithoutImprovement = 0
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Main training loop
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (!hasVal || epochsWithoutImprovement < maxEpochsVal)
        # Update weights using full-batch gradient descent
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)

        numEpoch += 1

        # Training loss tracking 
        trainingLoss = loss(ann, inputs', targets')
        push!(trainingLosses, trainingLoss)

        # Validation loss tracking and early stopping snapshot
        if hasVal
            valLoss = loss(ann, validationInputs', validationTargets')
            push!(validationLosses, valLoss)

            if valLoss < bestLoss
                bestLoss = valLoss
                bestANN = deepcopy(ann)
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end
        end

        # Record test loss for visualization purposes
        if hasTest
            testLoss = loss(ann, testInputs', testTargets')
            push!(testLosses, testLoss)
        end

        # Display progress only when requested
        showText && println("Epoch ", numEpoch, ": loss: ", trainingLoss)
    end

    # Return the best model snapshot if validation was used. Otherwise, the final model
    returnedANN = hasVal ? bestANN : ann
    return (returnedANN, trainingLosses, validationLosses, testLosses)
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "outputs and targets must have the same length"
    
    total = length(targets)
    
    # Calculate with logic vectorized operators
    TN = count(.!targets .& .!outputs)
    FP = count(.!targets .& outputs)
    FN = count(targets .& .!outputs)
    TP = count(targets .& outputs)
    
    # Confussion matrix as an Array{Int64,2}
    confMatrix = Int64[TN FP; FN TP]
    
    # Global metrics
    acc = (TN + TP) / total
    err = (FP + FN) / total
    
    # Particulas cases
    if TN == total
        sensitivity = 1.0
        specificity = 1.0
        ppv         = 1.0
        npv         = 1.0
    elseif TP == total
        sensitivity = 1.0
        specificity = 1.0
        ppv         = 1.0
        npv         = 1.0
    else
        sensitivity = (TP + FN == 0) ? 0.0 : TP / (TP + FN)
        specificity = (TN + FP == 0) ? 0.0 : TN / (TN + FP)
        ppv         = (TP + FP == 0) ? 0.0 : TP / (TP + FP)
        npv         = (TN + FN == 0) ? 0.0 : TN / (TN + FN)
    end
    
    # F-score 
    if (sensitivity + ppv == 0.0)
        f_score = 0.0
    else
        f_score = 2 * (sensitivity * ppv) / (sensitivity + ppv)
    end
    
    return (acc, err, sensitivity, specificity, ppv, npv, f_score, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "outputs and targets must have the same dimensions"
    numClasses = size(targets, 2)
    @assert numClasses != 2 "Two-column matrices are not valid"

    # If there is only one column, we use the function defined in the notebook 4.1
    if numClasses == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end

    # 1. Create the vectors to store the metrics
    sensitivities = zeros(Float64, numClasses)
    specificities = zeros(Float64, numClasses)
    ppvs          = zeros(Float64, numClasses)
    npvs          = zeros(Float64, numClasses)
    f1s           = zeros(Float64, numClasses)

    # 2. One-Against-All
    for c in 1:numClasses
        (_, _, sens, spec, ppv, npv, f1, _) = confusionMatrix(outputs[:, c], targets[:, c])
        sensitivities[c] = sens
        specificities[c] = spec
        ppvs[c]          = ppv
        npvs[c]          = npv
        f1s[c]           = f1
    end

    # 3. Confusion matrix
    confMatrix = [count(targets[:, i] .& outputs[:, j]) for i in 1:numClasses, j in 1:numClasses]

    # 4. Aggregation with weighted and macro strategies
    if weighted
        weights = vec(sum(targets, dims=1))
        totalInstances = sum(weights)
        if totalInstances == 0
            sensitivity = 0.0
            specificity = 0.0
            ppv         = 0.0
            npv         = 0.0
            f1          = 0.0
        else
            sensitivity = sum(sensitivities .* weights) / totalInstances
            specificity = sum(specificities .* weights) / totalInstances
            ppv         = sum(ppvs .* weights) / totalInstances
            npv         = sum(npvs .* weights) / totalInstances
            f1          = sum(f1s .* weights) / totalInstances
        end
    else
        sensitivity = mean(sensitivities)
        specificity = mean(specificities)
        ppv         = mean(ppvs)
        npv         = mean(npvs)
        f1          = mean(f1s)
    end

    # 5. Global accuracy and error rate
    acc = accuracy(outputs, targets)
    err = 1.0 - acc

    return (acc, err, sensitivity, specificity, ppv, npv, f1, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    booleanOutputs = classifyOutputs(outputs; threshold=threshold) 
    return confusionMatrix(booleanOutputs, targets; weighted=weighted)
end

