#! /usr/bin/env python

"""
Computes segmentation evaluation metrics for an input segmentation and its
corresponding ground truth. Comparisons can be in one of four forms:

    1. Single image probability map vs. single image ground truth
    2. Single image binary segmentation vs. single image binary ground truth
    3. Probability map stack vs. ground truth stack
    4. Binary segmentation stack vs. binary ground truth stack

References
----------

[1] Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition
     letters, 27(8), 861-874.
[2] Celebi, M. E., Schaefer, G., Iyatomi, H., Stoecker, W. V., Malters,
     J. M., & Grichnik, J. M. (2009). An improved objective evaluation
     measure for border detection in dermoscopy images. Skin Research and
     Technology, 15(4), 444-450.
[3] Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to
     ROC, informedness, markedness & correlation. Journal of Machine Learning
     Technologies, 2(1), 37-63.
[4] Seyedhosseini, M., Sajjadi, M., & Tasdizen, T. (2013). Image Segmentation
     with Cascaded Hierarchical Models and Logistic Disjunctive Normal 
     Networks. Computer Vision.
"""

import os
import glob
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from optparse import OptionParser
from subprocess import Popen, call, PIPE
from sys import stderr, exit, argv

def parse_args():
    global p
    p = OptionParser(usage = "%prog [options] probmap.png groundtruth.png",
                     epilog =
                     "Example: %s probmap.png groundtruth.png"
                     % os.path.basename(argv[0]))
    p.add_option("--output", dest = "path_out", metavar = "PATH",
                 help = "Output path to save stats to. (DEFAULT = Current "
                        "working directory).")
    p.add_option("--noroc", action = "store_true", dest = "noroc",
                 help = "Turns off the writing of ROC curves.")
    p.add_option("--nopr", action = "store_true", dest = "nopr",
                 help = "Turns off the writing of precision-recall curves.")
    p.add_option("--noint", action = "store_true", dest = "noint",
                 help = "Turns off writing of intermediate curves for all "
                        "images. Will only plot and save the summed curves.")
    p.add_option("--noimg", action = "store_true", dest = "noimg",
                 help = "Turns off writing of all plots and images.")
    p.add_option("--randomline", action = "store_true", dest = "randline",
                 help = "Adds a line equivalent to making a random choice "
                        "to all ROC plots.")
    (opts, args) = p.parse_args()
    pm_in, gt_in = check_args(args)
    path_out = check_opts(opts)
    return opts, pm_in, gt_in, path_out

def check_args(args):
    if len(args) is not 2:
        usage('Improper number of arguments.')
    pm_in = args[0]
    gt_in = args[1]
    return pm_in, gt_in

def check_opts(opts):
    # Set output path to current working directory, if not supplied
    if not opts.path_out:
        opts.path_out = os.getcwd()

    # Check that desired output path exists
    if not os.path.isdir(opts.path_out):
        usage("The output path {0} does not exist".format(opts.path_out))

    # Check new directory within output path, named segjudge-output, to store
    # all output files to. If this path already exists, append a number to the
    # end of the name.
    path_out = os.path.join(opts.path_out, "segjudge-output")
    if os.path.isdir(path_out):
        outdirs = sorted(glob.glob(path_out + '-*'))
        if not outdirs:
            nout = 2
        else:
            nout = int(outdirs[-1].split('-')[-1]) + 1 
        path_out = path_out + '-{0}'.format(nout)
    return path_out

def usage(errstr):
    print ""
    print "ERROR: %s" % errstr
    print ""
    p.print_help()
    print ""
    exit(1)

def writeROCplot(FPR, TPR, fnamein, fnameout, opts):
    plt.plot(FPR, TPR, linewidth = 2, label = "ROC")
    if opts.randline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle = "dashed", color = "red", linewidth = 2,
                 label = "Random")
    plt.xlabel("False Positive Rate", fontsize = 14)
    plt.ylabel("True Positive Rate", fontsize = 14)
    plt.title("ROC Curve - {0}".format(fnamein, fontsize = 14))
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize = 10, loc = 4)
    plt.tight_layout()
    plt.savefig(fnameout)
    plt.close()

def writePRplot(precision, recall, fnamein, fnameout, opts):
    plt.plot(precision, recall, linewidth = 2, label = "Precision")
    plt.xlabel("Precision", fontsize = 14)
    plt.ylabel("Recall", fontsize = 14)
    plt.title("Precision-Recall Curve - {0}".format(fnamein, fontsize = 14))
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(fnameout)
    plt.close()

def computeMetrics(FP, FN, TP, TN, thresh):
    # Compute evaluation metrics 
    with np.errstate(invalid = "ignore"):
        N = FP + FN + TP + TN
        FPR = np.divide(FP, FP + TN)
        TPR = np.divide(TP, TP + FN)
        FNR = np.divide(FN, FN + TP)
        TNR = np.divide(TN, TN + FP)
        precision = np.divide(TP, TP + FP) # Ref [1]
        recall = TPR # Ref [1]
        accuracy = np.divide(TP + TN, N) # Ref [1]
        errorprob = np.divide(FP + FN, N) # Ref [2]
        fvalue = np.divide(2 * TP, 2 * TP + FP + FN) # Ref [1]
        jaccard = np.divide(TP, FP + TP + FN) # Ref [3]
        gmean = math.sqrt(recall*TNR) # Ref [4]
        ppv = precision # Ref [1]
        npv = np.divide(TN, TN + FN) # Ref [1]
        fdr = np.divide(FP, FP + TP) # Ref [1]
        sensitivity = TPR # Ref [1]
        specificity = TNR # Ref [1]
        informedness = sensitivity + specificity - 1 # Ref [1]
        markedness = precision + npv - 1 # Ref [1]
        MCC = np.divide(TP*TN-FP*FN,
                        math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    # Create data string
    csvdata = "%d,%d,%d,%d,%d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f," \
              "%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f," \
              "%0.5f,%0.5f,%0.5f,%0.5f,%0.5f" % (thresh, FP, FN, TP, TN,
              FPR, FNR, TPR, TNR, precision, recall, accuracy,
              errorprob, fvalue, jaccard, gmean, ppv, npv, fdr,
              sensitivity, specificity, informedness, markedness, MCC)
    return TPR, FPR, precision, recall, csvdata


if __name__ == "__main__":

    opts, pm_in, gt_in, path_out = parse_args()

    # Create main output path
    os.makedirs(path_out)

    # Create sub-directories in output path for various outputs, as needed
    path_csv = os.path.join(path_out, "csv")
    os.makedirs(path_csv)
 
    path_roc = os.path.join(path_out, "roc")
    os.makedirs(path_roc)

    path_auc = os.path.join(path_roc, "auc")
    os.makedirs(path_auc)

    if not opts.noroc:
        path_rocplots = os.path.join(path_roc, "plots")
        os.makedirs(path_rocplots)

    if not opts.nopr:
        path_pr = os.path.join(path_out, "prec_recall")
        os.makedirs(path_pr)

    # Check if input arguments are directories or single image files
    if os.path.isdir(pm_in):
        pm_files = sorted(glob.glob("{0}/*".format(pm_in)))
    else:
        pm_files = [pm_in]

    if os.path.isdir(gt_in):
        gt_files = sorted(glob.glob("{0}/*".format(gt_in)))
    else:
        gt_files = [gt_in]

    if len(pm_files) != len(gt_files):
        usage("The number of probability map images does not equal the "
              "number of ground truth images.")
    nfiles = len(pm_files)

    csvheader = "\"Threshold\"," \
                "\"FP\"," \
                "\"FN\"," \
                "\"TP\"," \
                "\"TN\"," \
                "\"FPR\"," \
                "\"FNR\"," \
                "\"TPR\"," \
                "\"TNR\"," \
                "\"Precision\"," \
                "\"Recall\"," \
                "\"Accuracy\"," \
                "\"Error Probability\"," \
                "\"F-value\"," \
                "\"Jaccard Similarity\"," \
                "\"G-Mean\"," \
                "\"Positive Predictive Value\"," \
                "\"Negative Predictive Value\"," \
                "\"False Discovery Rate\"," \
                "\"Sensitivity\"," \
                "\"Specificity\"," \
                "\"Informedness\"," \
                "\"Markedness\"," \
                "\"Matthews Correlation Coefficient\""

    FParray = np.zeros([nfiles, 256], dtype = "f")
    FNarray = np.zeros([nfiles, 256], dtype = "f")
    TParray = np.zeros([nfiles, 256], dtype = "f")
    TNarray = np.zeros([nfiles, 256], dtype = "f")
    imgtype = []

    for i in range(0, nfiles):
        print "Analyzing {0}...".format(pm_files[i])
        TPRlist = []
        FPRlist = []
        preclist = []
        reclist = []
        bname = os.path.splitext(os.path.basename(pm_files[i]))[0]

        pmi = misc.imread(pm_files[i])
        gti = misc.imread(gt_files[i])
        if (pmi.shape[0] != gti.shape[0]) or (pmi.shape[1] != gti.shape[1]):
            usage("The images {0} and {1} do not have the same "
                  "dimensions".format(pm_files[i], gt_files[i]))

        # Check if the input image is binary or not. If it is binary, compute
        # metrics at only one threshold (pix values > 0). If it is not binary
        # (i.e. it is a probability map), then threshold at all intensity
        # values and compute metrics.
        if len(np.unique(pmi)) <= 2:
            intensities = [0]
            imgtype.append(0)
            fname = os.path.join(path_csv, bname + ".csv")
        else:
            intensities = range(0, 256)
            imgtype.append(1)
            fname = os.path.join(path_csv, bname + ".csv")
       
        # Open a new csv file if intermediate files are desired 
        if not opts.noint:
            h = open(fname, "a+")

        # Exit if there are different types of images in the input direcoty.
        # Must have either all binary images or all grayscale images.
        if i != 0 and imgtype[i] != imgtype[i-1]:
            shutil.rmtree(path_out)
            usage("Images must be either all binary or all grayscale.")

        for j in range(0, len(intensities)):
            # Threshold the image at the given intensity level
            threshi = np.where(pmi > intensities[j], 1, 0)

            # Compute the confusion matrix
            FP = np.float(np.sum((threshi == 1) & (gti == 0)))
            FN = np.float(np.sum((threshi == 0) & (gti == 1)))
            TP = np.float(np.sum((threshi == 1) & (gti == 1)))
            TN = np.float(np.sum((threshi == 0) & (gti == 0)))

            # Get metrics
            TPR, FPR, precision, recall, data = computeMetrics(FP, FN, TP, TN, j)
            FParray[i, j] = FP
            FNarray[i, j] = FN
            TParray[i, j] = TP
            TNarray[i, j] = TN

            if imgtype[i] == 1: 

                TPRlist.append(TPR)
                FPRlist.append(FPR)
                preclist.append(precision)
                reclist.append(recall)

                if not opts.noint:
                    if j == 0:
                        h.write(csvheader + "\n")
                
                    h.write(data + "\n")

                    if j == len(intensities)-1:
                        h.close()
            else:
                if not opts.noint:
                    h.write(csvheader + "\n")
                    h.write(data + "\n")

        # Save single image plots of ROC and precision-recall if desired
        if imgtype[i] == 1 and not opts.noint:
            auc = np.absolute(np.trapz(TPRlist, FPRlist))
            np.savetxt(os.path.join(path_auc, bname.split(".")[0] + ".txt"),
                np.array([auc]), fmt = "%0.5f")
            if not opts.noimg  and not opts.noroc:
                fnameout = os.path.join(path_rocplots, "{0}.png".format(
                                        bname.split(".")[0]))
                writeROCplot(FPRlist, TPRlist, bname, fnameout, opts)
            if not opts.noimg and not opts.nopr:
                fnameout = os.path.join(path_pr, "{0}.png".format(
                                        bname.split(".")[0]))
                writePRplot(preclist, reclist, bname, fnameout, opts)

    # If more than one file was input, calculate and report summed metrics and
    # plots across the entire stack of images. If the images are grayscale,
    # calculate composite metrics at each threshold value.

    if len(pm_files) == 1:
       #os.rmdir(path_roc)
       #os.rmdir(path_pr)
       exit(0)

    # Compute sum of confusion matrix values for all images at each threshold
    FPsum = np.sum(FParray, 0)
    FNsum = np.sum(FNarray, 0)
    TPsum = np.sum(TParray, 0)
    TNsum = np.sum(TNarray, 0)

    # Open sum csv file
    fname = os.path.join(path_csv, "sum.csv")
    h = open(fname, "a+")
    h.write(csvheader + "\n")

    if imgtype[0] == 1: 
        # Compute summed metrics at each threshold
        TPRsum = []
        FPRsum = []
        precsum = []
        recsum = []
        for j in range(0, len(intensities)):
            TPRsumj, FPRsumj, precsumj, recsumj, datasumj = computeMetrics(
                FPsum[j], FNsum[j], TPsum[j], TNsum[j], j)
            TPRsum.append(TPRsumj)
            FPRsum.append(FPRsumj)
            precsum.append(precsumj)
            recsum.append(recsumj)
            h.write(datasumj + "\n")
        h.close()

        # Compute area under the ROC curve (AUC) for the summed curve
        aucsum = np.absolute(np.trapz(TPRsum, FPRsum))
        np.savetxt(os.path.join(path_auc, "sum.txt"), np.array([aucsum]),
            fmt = "%0.5f")

        # Plot and save composite ROC and precision-recall curves, if desired
        if not opts.noimg and not opts.noroc:
            fnameout = os.path.join(path_rocplots, "sum.png")
            writeROCplot(FPRsum, TPRsum, "sum", fnameout, opts)
        if not opts.noimg and not opts.nopr:
            fnameout = os.path.join(path_pr, "sum.png")
            writePRplot(precsum, recsum, "sum", fnameout, opts)
    else:
        TPRsum, FPRsum, precsum, recsum, datasum = computeMetrics(FPsum[0],
            FNsum[0], TPsum[0], TNsum[0], 0)
        h.write(datasum + "\n")
        h.close()
        os.rmdir(path_roc)
        os.rmdir(path_pr)
