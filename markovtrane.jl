## Usings
using MIDI
using MusicManipulations
using PyPlot
using Printf
using StatsBase
using Random
using SparseArrays
#using GraphMakie
#using CairoMakie
#using GraphPlot
MIDIMAX = 128

#####################################################################
## Params
#####################################################################

mididir     = joinpath(pwd(),"repertoire")
outpath     = joinpath(pwd(),"out.mid")
order       = 3
outlen      = 50
firstnotes  = [73,76]

firstvels   = false     # comment this line
#firstvels   = [100,109]  # and uncomment this line to experiment with velocity inference.

#####################################################################
## Import Data
#####################################################################

inpaths = joinpath.(mididir,readdir(mididir))
midipaths = inpaths[findall(endswith(".mid"),inpaths)]
numfiles        = size(midipaths)[1]
mididata        = Any[]
midi_noteon     = Any[]
midi_noteoff    = Any[]

for ff = 1:numfiles
    Printf.@printf("Opening f%d: %s\n",ff,midipaths[ff])
    infile = MIDI.readMIDIFile(midipaths[ff])
    # [dur, on/off, note, vel]
    fidxs = findall(typeof.(infile.tracks[1].events).==MIDIEvent)
    indata = infile.tracks[1].events[fidxs] # MIDI events only
    indata2 = zeros(Int64,size(fidxs)[1],4); # [time, status, data1, data2]
    for ii = 1:size(fidxs)[1]
        indata2[ii,:] = [
            indata[ii].dT, 
            indata[ii].status, 
            indata[ii].data[1], 
            indata[ii].data[2]]
    end

    push!(mididata,indata2) # push midi data
    push!(midi_noteon, indata2[indata2[:,2].==144,:]) # noteon's only
    push!(midi_noteoff, indata2[indata2[:,2].==128,:]) # noteoff's only
end

#####################################################################
## Markov Model Function Defs
#####################################################################

# Build N-grams
primitive = 1:MIDIMAX # just a range...

function make_ngramdict(primitive, order)
    primlen = size(primitive)[1]
    seqdict = repeat(primitive,inner=primlen^(order-3),outer=primlen^(order-1))
    for ord = 2:order
        seqtemp = repeat(primitive, inner=primlen^(ord-1), outer=primlen^(order-ord))
        seqdict = [seqdict seqtemp]
    end
    seqdict = [seqdict zeros(Int64, size(seqdict)[1])] # counts
    return seqdict
end

function makemarkov(markovin, seq_in, order)
    runs = seq_in[1:end-order+1]
    for ord = 2:order # collect all runs (sequences to query)
        runtemp = seq_in[ord:end-order+ord]
        runs = [runs runtemp]
    end
    runs = [CartesianIndex(Tuple(rr)) for rr in eachrow(runs)]

    [markovin[rr]+=1 for rr in runs] # increment for each occurrence of run

    return markovin
end

# Construct 2D Markov Matrix (lookup table)
function makemarkov2D(markov2D, seq_in, order)
    runs = seq_in[1:end-order+1]
    for ord = 2:order # collect all runs (sequences to query)
        runtemp = seq_in[ord:end-order+ord]
        runs = [runs runtemp]
    end

    for rr in eachrow(runs)
        queryidx = findperm(rr)
        markov2D[queryidx,4] += 1
    end
    
    return markov2D
end

findperm(query_notes) = sum(query_notes .* MIDIMAX.^(0:order-1) .- MIDIMAX.^(0:order-1))+1

#####################################################################
## Construct Markov Models
#####################################################################
markov_notes     = zeros(MIDIMAX,MIDIMAX,MIDIMAX)
markov_vels      = zeros(MIDIMAX,MIDIMAX,MIDIMAX)
markov2D_notes   = make_ngramdict(1:MIDIMAX,order)

notetotal = 0
for ff = 1:numfiles
    maxidx = size(midi_noteon[ff])[1]

    #markov_times = makemarkov(markov_times, midi_noteon[ff][:,1], order)
    global markov_notes     = makemarkov(markov_notes, midi_noteon[ff][:,3], order)
    global markov_vels      = makemarkov(markov_vels, midi_noteon[ff][:,4], order)
    global markov2D_notes   = makemarkov2D(markov2D_notes, midi_noteon[ff][:,3], order)
    
    global notetotal += maxidx
    Printf.@printf("Analyzed f%d (%d nts)\n",ff,maxidx)
end

# build submatrix
#slice_xy = dropdims(sum(markov_notes,dims=3),dims=3)
#slice_yz = dropdims(sum(markov_notes,dims=1),dims=1)
#slice_zx = dropdims(sum(markov_notes,dims=2),dims=2)
#slice_zx = permutedims(slice_zx,[2,1]) # more intuitive...

# collapses
function markovcollapse(markovin, collapsedim, prenorm_transpose=false)
    sliced = dropdims(sum(markovin,dims=collapsedim),dims=collapsedim)
    sliced = prenorm_transpose ? permutedums(sliced,[2,1]) : sliced
    rowsums = sum(sliced,dims=2)
    rowsums[rowsums.==0] .= 1 # prime zero-sum rows with 1 for 'noise'...
    sliced_norm = sliced ./ repeat(rowsums,outer=[1,size(sliced)[2]])
    return sliced, sliced_norm
end

function normalizerows(mtxin)
    rowsums = sum(mtxin,dims=2)
    rowsums[rowsums.==0] .= 1
    return mtxin ./ repeat(rowsums,outer=[1,size(mtxin)[2]])
end

slice_xy, slice_xy_norm = markovcollapse(markov_notes,3)
slice_xy, slice_yz_norm = markovcollapse(markov_notes,1)
slice_xy, slice_zx_norm = markovcollapse(markov_notes,2)

#slice_xy_norm = normalizerows(slice_xy)
#slice_yz_norm = normalizerows(slice_yz)
#slice_zx_norm = normalizerows(slice_zx)

#####################################################################
## Make Markov Inference and Compose Solo
#####################################################################

out_notes = zeros(Int64,outlen,1)
out_vels = zeros(Int64,outlen,1)
out_notes[1:2] = firstnotes
out_vels[1:2] = firstvels==false ? [64 64] : firstvels

rng = MersenneTwister(10)

function markov_infer(range, valsin, markovin)
    dist = markovin[valsin[1], valsin[2],:]
    dist ./= sum(dist)
    return sample(rng, 1:MIDIMAX, ProbabilityWeights(dist))
end

for nn = 3:size(out_notes)[1]
    out_notes[nn] = markov_infer(1:MIDIMAX, out_notes[(nn-2):(nn-1)], markov_notes)
    out_vels[nn] = firstvels==false ? 64 : markov_infer(1:MIDIMAX, out_vels[(nn-2):(nn-1)], markov_vels)
end

out_times = ones(Int,outlen*2,2)
out_times[1:2:end] .= 100 # note on dT
out_times[2:2:end] .= 10 # note off dT (quick)

out_status = zeros(Int,outlen*2,2)
out_status[1:2:end] .= 0x90 # note on message
out_status[2:2:end] .= 0x80 # note off message

out_notedata = repeat([out_notes out_vels], inner=[2,1]) # note on, off

outfile = MIDI.readMIDIFile(inpaths[1]) # open an input file to use file structure
outfilelen = size(outfile.tracks[1].events,1)

# fill MIDI file with inferred notes
for nn = 1:outlen*2
    outfile.tracks[1].events[4+nn].dT       = out_times[nn]
    outfile.tracks[1].events[4+nn].status   = out_status[nn]
    outfile.tracks[1].events[4+nn].data[1]  = out_notedata[nn,1]
    outfile.tracks[1].events[4+nn].data[2]  = out_notedata[nn,2]
end

# fill remainder of MIDI file with blank data
for nn = (outlen*2+1):outfilelen-4
    outfile.tracks[1].events[4+nn].dT       = 1
    outfile.tracks[1].events[4+nn].status   = out_status[end]
    outfile.tracks[1].events[4+nn].data[1]  = out_notedata[end,1]
    outfile.tracks[1].events[4+nn].data[2]  = out_notedata[end,2]
end

writeMIDIFile(outpath, outfile)
Printf.@printf("MIDI output written to: %s\n", outpath)

#####################################################################
# Plots
#####################################################################

## Distribution of Collected N-Grams
fig1 = PyPlot.figure()
PyPlot.plot(1:size(markov2D_notes)[1], markov2D_notes[:,4])
display(fig1)
PyPlot.ylabel("Occurrences")
PyPlot.xlabel("Sequence Index (flattened)")
PyPlot.title("N-Gram Summary")
display(fig1)

## Plot 3D Slice (unnormalized)
fig2 = PyPlot.figure()
ax = Axes3D(fig2)
PyPlot.plot_surface(1:MIDIMAX, 1:MIDIMAX, slice_xy, cmap="inferno")
ax.view_init(elev=40,azim=250)
ax.w_xaxis.set_pane_color((.30, .30, .30, 1))
ax.w_yaxis.set_pane_color((.30, .30, .30, 1))
ax.w_zaxis.set_pane_color((.30, .30, .30, 1))
PyPlot.xlim((0,MIDIMAX))
PyPlot.ylim((0,MIDIMAX))
PyPlot.xlabel("Start Note")
PyPlot.ylabel("End Note")
PyPlot.title("Note 1-2 (Histogram)")
display(fig2)

## Plot 3D Slice (normalized)
fig3 = PyPlot.figure()
ax = Axes3D(fig3)
PyPlot.plot_surface(1:MIDIMAX, 1:MIDIMAX, slice_xy_norm, cmap="inferno")
ax.view_init(elev=40,azim=250)
ax.w_xaxis.set_pane_color((.30, .30, .30, 1))
ax.w_yaxis.set_pane_color((.30, .30, .30, 1))
ax.w_zaxis.set_pane_color((.30, .30, .30, 1))
PyPlot.xlim((0,MIDIMAX))
PyPlot.ylim((0,MIDIMAX))
PyPlot.xlabel("Start Note")
PyPlot.ylabel("End Note")
PyPlot.title("Note 1-2 (PMF Normalized)")
display(fig3)

## Scatterplot 4D
sparseseqs = markov2D_notes[findall(!iszero, markov2D_notes[:,4]),:]

fig4 = PyPlot.figure()
ax = Axes3D(fig4)
ax.scatter(sparseseqs[:,1], sparseseqs[:,2], sparseseqs[:,3], c=sparseseqs[:,4]./maximum(sparseseqs[:,4]), edgecolors="none", marker=".", alpha=0.4, cmap="hot")
#ax.w_xaxis.set_pane_color((0, 0, 0, 1))
#ax.w_yaxis.set_pane_color((0, 0, 0, 1))
#ax.w_zaxis.set_pane_color((0, 0, 0, 1))
ax.view_init(elev=20,azim=260)
plt.xlabel("Note 1")
plt.ylabel("Note 2")
plt.title("Note 1-2-3 (Histogram)")
display(fig4)