## Usings
using MIDI
using MusicManipulations
using PyPlot
using Printf
const plt = PyPlot

MIDIMAX = 128

##

indir = joinpath(pwd(),"markov")
inpaths = joinpath.(indir,readdir(indir))
midipaths = inpaths[findall(endswith(".mid"),inpaths)]

## Read first four messages (TODO: interpret 2,3,4)
#=
ii=1
Printf.@printf("Event %d: (%c)\n", ii, infile.tracks[1].events[ii].metatype)
for jj = 1:size(infile.tracks[1].events[ii].data)[1]
    Printf.@printf("%c ",infile.tracks[1].events[ii].data[jj])
end
Printf.@printf("\n")

for ii = 2:4
    Printf.@printf("Event %d (%c)\n",ii, infile.tracks[1].events[ii].metatype)
    for jj = 1:size(infile.tracks[1].events[ii].data)[1]
        Printf.@printf("%d ",infile.tracks[1].events[ii].data[jj])
    end
    Printf.@printf("\n")
end
=#

## Convert data into array

numfiles        = size(midipaths)[1]
mididata        = Any[]
midi_noteon     = Any[]
midi_noteoff    = Any[]

for ff = 1:numfiles
    Printf.@printf("Opening f%d: %s\n",ff,inpaths[ff])
    infile = MIDI.readMIDIFile(inpaths[ff])
    # [dur, on/off, note, vel]
    fidxs = findall(x->typeof(x)==MIDIEvent,infile.tracks[1].events)
    indata = infile.tracks[1].events[fidxs]
    indata2 = zeros(Int64,size(fidxs)[1],4); 
#    for ii = 5:size(infile.tracks[1].events)[1]
#        indata[ii-4,:] = [
#            infile.tracks[1].events[ii].dT, 
#            infile.tracks[1].events[ii].status, 
#            infile.tracks[1].events[ii].data[1], 
#            infile.tracks[1].events[ii].data[2]]
#    end
    for ii = 1:size(fidxs)[1]
        indata2[ii,:] = [
            indata[ii].dT, 
            indata[ii].status, 
            indata[ii].data[1], 
            indata[ii].data[2]]
    end

    push!(mididata,indata2) # push midi data
    push!(midi_noteon,indata2[findall(x -> x==144, indata2[:,2]),:])
    push!(midi_noteoff,indata2[findall(x -> x==128, indata2[:,2]),:])
end

## build markov matrix

## show all possible sequences
order = 3

prim = 1:MIDIMAX

seqs = repeat(prim,inner=MIDIMAX^(order-3),outer=MIDIMAX^(order-1))
for ord = 2:order
    seqtemp = repeat(prim, inner=MIDIMAX^(ord-1), outer=MIDIMAX^(order-ord))
    seqs = [seqs seqtemp]
end
seqs = [seqs zeros(Int64, size(seqs)[1])]

#findperm(notes) = notes[1] + notes[2]*MIDIMAX^1 - MIDIMAX^1 + notes[3]*MIDIMAX^2 - MIDIMAX^2
findperm(notes) = sum(notes .* MIDIMAX.^(0:order-1) .- MIDIMAX.^(0:order-1))+1

##
markovmtx = zeros(MIDIMAX,MIDIMAX)
notetotal = 0


for ff = 1:numfiles
    notevals = midi_noteon[ff][:,3] # noteon data only
    maxidx = size(notevals)[1]
    runs = notevals[1:end-order+1]
    for ord = 2:order # collect all runs (sequences to query)
        runtemp = notevals[ord:end-order+ord]
        runs = [runs runtemp]
    end

    for ii = 1:size(runs)[1]
        queryidx = findperm(runs[ii,:])
        seqs[queryidx,4] += 1 
    end
    #=
    for nn = 1:MIDIMAX
        noteidxs = findall(x->x==nn, notevals) # find a given note
        nextnotes = testset[(noteidxs.+1)[findall(x -> x<maxidx, (noteidxs .+ 1))]] # find succeeding note
        for iter = 1:size(nextnotes)[1]
            markovmtx[nn,nextnotes[iter]] += 1;
        end
    end
    =#

    notetotal += maxidx
    Printf.@printf("Analyzed f%d (%d nts)\n",ff,maxidx)
end

## build submatrix


## normalize rows
#=
sums = sum(markovmtx, dims=2)
valididxs = getindex.(findall(x->x>1, sums),1)
validsums = repeat(sums[valididxs], outer=[1,MIDIMAX])
markovmtx[valididxs,:] = markovmtx[valididxs,:]./validsums
=#

## plot
#=
fig1 = PyPlot.figure()
ax = Axes3D(fig1)
PyPlot.plot_surface(1:MIDIMAX,1:MIDIMAX,markovmtx, cmap="inferno")
ax.view_init(elev=80,azim=180)
plt.xlabel("Start Note")
plt.ylabel("End Note")
display(fig1)
=#

##

fig2 = PyPlot.figure()
PyPlot.plot(1:size(seqs)[1], seqs[:,4])
display(fig2)