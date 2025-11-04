# copy from https://gitlab.com/RutgersLBSR/fe-toolkit
# d2177aee1e8fd2de57fcbbf1798326d47623448c
# with adjustment to accept batter formatted inputs
if __name__ == "__main__":

    import copy
    import sys
    import argparse
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    from fetkutils import SimpleScheduleOpt
    from fetkutils import SimpleScheduleRead

    parser = argparse.ArgumentParser \
    ( formatter_class=argparse.RawDescriptionHelpFormatter,
      description="""
      Perform PSO, AR, or KL analysis on a schedule that is either
      read from a file or optimized.

      PSO is the phase space overlap index.
      AR is a predicted replica exchange acceptance ratio.
      KL is the symmetrized Kullback-Leibler divergence

      To perform the analysis, the simulation results from a short burn-in
      is read. That is, this script reads the dvdl_*.dat and 
      efep_*_*.dat files generated from edgembar-amber2dats.py.

      Example 1
      ---------
      Optimize as 12-point schedule to minimize the PSO variance.
      In this example, there are 10 parameters (the endpoints are
      fixed to lambda=0 and lambda=1, and the remaining values are
      optimized without constraints).

      fetkutils-tischedule.py --opt=12 --pso --out=pso_12.txt \\
                    --plot=pso_12.png ./dats

      Example 2
      ---------
      Optimize as 12-point schedule to minimize the RE variance.
      In this example, there are 10 parameters (the endpoints are
      fixed to lambda=0 and lambda=1, and the remaining values are
      optimized without constraints).

      fetkutils-tischedule.py --opt=12 --ar --out=ar_12.txt \\
                    --plot=ar_12.png ./dats


      Example 3
      ---------
      Optimize a symmetric 12-point schedule to minimize the AR variance.
      In this example, there are 5 parameters (the endpoints are
      fixed to lambda=0 and lambda=1, and the lambda>0.5 values are
      determined fro mthe lambda<0.5 values).

      fetkutils-tischedule.py --opt=12 --ar --sym --out=ar_sym_12.txt \\
                    --plot=ar_sym_12.png ./dats


      Example 4
      ---------
      Optimize a 12-point SSC(alpha) schedule to minimize the AR variance.
      In this example, there is only 1 parameter, the SSC alpha parameter,
      which can vary from 0 to 2.

      fetkutils-tischedule.py --opt=12 --ar --ssc --sym \\
                    --out=ar_sscsym_12.txt \\
                    --plot=ar_sscsym_12.png ./dats


      Example 5
      ---------
      Optimize a 12-point asymmetric SSC(alpha) schedule to minimize the 
      AR variance. In this example, there are 2 parameters; the alpha
      parameter is lambda-dependent and scaled from alpha0 to alpha1.

      fetkutils-tischedule.py --opt=12 --ar --ssc --out=ar_ssc_12.txt \\
                    --plot=ar_ssc_12.png ./dats


      Example 6
      ---------
      Construct a 12-point asymmetric SSC(alpha) schedule, given the
      parameters.  This does not perform an optimization.

      fetkutils-tischedule.py --read=12 --ar --ssc \\
                    --alpha0=1.5 --alpha1=1.6 \\
                    --out=ar_ssc_12_1.5_1.6.txt \\
                    --plot=ar_ssc_12.png ./dats

      Example 7
      ---------
      Read a schedule from a file. This does not perform an optimization.

      fetkutils-tischedule.py --read=ar_ssc_12_1.5_1.6.txt --ar \\
                    --plot=ar_ssc_12_1.5_1.6.png ./dats



""" )

    parser.add_argument \
        ('directory',
         metavar='directory',
         type=str,
         help="the directory containing the dvdl_*.dat and "+
         "efep_*_*.dat files")

    parser.add_argument \
        ("--opt",
         type=int,
         help="Optimize a schedule. If this option is not used, then "+
         "--read must be used. The value of OPT is the integer number "+
         "lambda values in the schedule. If --varlim-minval > 0, then "+
         "the value of --opt is the minimum allowable number of "+
         "lambda values in the schedule",
         required=False)


    parser.add_argument \
        ("--varlam-multiple",
         type=int,
         help="The output schedule size produced from --varlam-minval "+
         "will be an integer multiple of --varlam-multiple (default: 4)",
         default=4,
         required=False)

    parser.add_argument \
        ("--varlam-minval",
         type=float,
         help="If --varlam-minval > 0, then optimize a schedule whose "+
         "size is given by --opt. If the minimum PSO/AR/KL is less than "+
         "the given minimum value, then increase the schedule by "+
         "--varlam-multiple.  The value of --opt must be divisible by "+
         " --varlam-multiple on input",
         default=0,
         required=False)

    

    
    parser.add_argument \
        ("--read",
         type=str,
         help="Read a schedule. If this option is not used, then --opt "+
         "must be used. If --ssc is not used, then the value of READ is "+
         "a filename. The file should contain 1 column; the rows are the "+
         "lambda values.  If --ssc is used, then the value of READ is "+
         "an integer: the number of lambda values. See also: --alpha0 "+
         "and --alpha1",
         required=False)

    parser.add_argument \
        ("--pso",
         action='store_true',
         help="Analyze/optimize based on the predicted phase space "+
         "overlap index")
    
    parser.add_argument \
        ("--ar",
         action='store_true',
         help="Analyze/optimize based on the predicted replica exchange "+
         "acceptance ratio")
    
    parser.add_argument \
        ("--kl",
         action='store_true',
         help="Analyze/optimize based on the predicted Kullback-Leibler "+
         "divergence of the deltaU distributions")
    
    parser.add_argument \
        ("--ssc",
         action='store_true',
         help="Restrict the optimization to the SSC(alpha) scheduling. "+
         "If --sym is not used, then there are two parameters: "+
         "alpha(lambda=0) and alpha(lambda=1). If --sym is present, "+
         "then there is only 1 parameter: a lambda-independent alpha.")

    parser.add_argument \
        ("--alpha0",
         type=float,
         default=2.0,
         help="If --read=N and --ssc are used, then ALPHA0 is the "+
         "SSC(alpha) parameter. If --alpha1 is also used, then alpha "+
         "is lambda-dependent. (Default: 2.0)")

    parser.add_argument \
        ("--alpha1",
         type=float,
         help="If --read=N and --ssc are used, then the schedule is "+
         "SSC(alpha(lambda)), where alpha is a lambda-dependent parameter "+
         "that scales from alpha0 to alpha1. If --alpha1 is unset, then "+
         "alpha is lambda-independent. (Default: None)")
    
    
    parser.add_argument \
        ("-o","--out",
         help="write schedule to specified file",
         type=str,
         required=False )
    

    parser.add_argument \
        ("--sym",
         help="if present, force the schedule to be symmetric around "+
         "lambda=0.5",
         action='store_true',
         required=False )

    parser.add_argument \
        ("--maxgap",
         help="maximum allowable gap between consecutive lambdas "+
         "(default: 1.0)",
         type=float,
         default=1.0,
         required=False )
    
    parser.add_argument \
        ("--spen",
         help="minimum allowable value before a penalty is applied "+
         "(default: 0.1)",
         type=float,
         default=0.1,
         required=False )

    parser.add_argument \
        ("--digits",
         help="round schedule to this number of digits right of the "+
         "decimal (default: 5)",
         type=float,
         default=5,
         required=False )
    
    parser.add_argument \
        ("--start",
         help="Exclude this percentage of simulation as equilibration "+
         "(range: 0. to 1., default: 0.)",
         type=float,
         default=0.,
         required=False )
    
    parser.add_argument \
        ("--stop",
         help="Exclude all simulation after this percentage (range: 0. "+
         "to 1., default: 1.)",
         type=float,
         default=1.,
         required=False )

    parser.add_argument \
        ("--clean",
         help="Modify the values to remove noise and ignore large overlaps "+
         "between distant lambdas. By using this option, one guarantees "+
         "exponential decay of the values when moving away from the diagonal.",
         action='store_true',
         required=False )

    parser.add_argument \
        ("-T","--temp",
         help="Temperature (K). (default: 298.)",
         type=float,
         default=298.,
         required=False )

    parser.add_argument \
        ("--plot",
         help="Write heatmap and path projection to an image file",
         type=str,
         required=False )

    
    try:
        import pkg_resources
        version = pkg_resources.require("fetkutils")[0].version
    except:
        version = "unknown"
    
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format\
                        (version=version))


    
    args = parser.parse_args()

    if args.opt is not None and args.read is not None:
        raise Exception("--opt and --read are mutually exclusive options")

    if args.pso + args.ar + args.kl > 1:
        raise Exception("--pso, --ar, --kl are mutually exclusive options")

    if not (args.pso or args.ar or args.kl):
        raise Exception("Either --pso, --ar, or --kl is required")


    
    if args.opt is not None:

        if args.varlam_minval > 0:
            if args.opt % args.varlam_multiple != 0:
                raise Exception(f"--opt ({args.opt}) is not a multiple of {args.varlam_multiple}")
            nlam = args.opt - args.varlam_multiple
            for it in range(20):
                nlam += args.varlam_multiple
                info, pedata, interp = SimpleScheduleOpt\
                    ( nlam,
                      args.directory, args.temp, args.start, args.stop,
                      args.ssc, args.sym, args.pso, args.ar, args.kl, args.clean,
                      args.maxgap, args.spen, args.digits, True )
                if info.minval > args.varlam_minval:
                    break
            if info.minval < args.varlam_minval:
                print(f"A schedule with size {nlam} produced a minval of {info.minval} which is less than the target {args.varlam_minval}")
            
        else:
            ################################################
            # optimize a schedule
            info, pedata, interp = SimpleScheduleOpt\
                ( args.opt,
                  args.directory, args.temp, args.start, args.stop,
                  args.ssc, args.sym, args.pso, args.ar, args.kl, args.clean,
                  args.maxgap, args.spen, args.digits, True )
    else:
        ################################################
        # Read or evaluate a schedule
        info, pedata, interp = SimpleScheduleRead\
            ( args.read,
              args.directory, args.temp, args.start, args.stop,
              args.ssc, args.alpha0, args.alpha1, args.pso, args.ar,
              args.kl, args.clean, args.digits, True )

        
    if args.out is not None:
        info.Save(args.out)
    

    if args.plot is not None:
        m = 151
        ls = np.linspace(0,1,m)
        mat = np.zeros( (m,m) )
        for i in range(m):
            for j in range(m):
                mat[i,j] = interp.InterpValueFromLambdas(ls[i],ls[j])

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches( 6.5, 2.9, forward=True)

        pad = 0.08
        fontsize = 8
        label = "PSO"
        if args.ar:
            label = "AR"
        elif args.kl:
            label = "exp(-KL)"
        
        fig.suptitle(f"Predicted {label}", fontsize=fontsize)
            
        for lam in info.lams:
            ax1.plot( [lam,lam],[0,1], c='r',ls=':', lw=1 )

        ax1.plot( info.midpts, info.vals, c='k', ls='-', lw=1 )
        ax1.scatter( info.midpts, info.vals, c='k', s=3 )

        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.xaxis.set_major_locator(MultipleLocator(0.2))
        ax1.set_ylim(0,1)
        ax1.set_xlim(-0.05,1.05)
        ax1.set_ylabel(label,fontsize=fontsize,labelpad=pad)
        ax1.set_xlabel(r"${\lambda}$",fontsize=fontsize,labelpad=pad)
        ax1.tick_params(axis='both', which='major',
                        labelsize=fontsize, pad=1.5)
        ax1.set_aspect('equal')
        
        im2 = ax2.imshow(mat,
                         cmap='bwr',
                         interpolation='nearest',
                         vmin=0,
                         vmax=min(2,np.amax(mat)),
                         origin='lower',
                         extent=[0,1,0,1])
        
        ax2.scatter( info.lams[:-1],info.lams[1:], c='k', s=3 )
        
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        
        ax2.yaxis.set_major_locator(MultipleLocator(0.2))
        ax2.xaxis.set_major_locator(MultipleLocator(0.2))
        ax2.set_xlabel(r"${\lambda_{1}}$",fontsize=fontsize,labelpad=pad)
        ax2.set_ylabel(r"${\lambda_{2}}$",fontsize=fontsize,labelpad=pad)
        for t in cax2.get_yticklabels():
            t.set_fontsize(fontsize)
        cax2.set_ylabel(label,fontsize=fontsize,labelpad=pad)
        cax2.tick_params(axis='both', which='major',
                         labelsize=fontsize, pad=1.5)
        ax2.tick_params(axis='both', which='major',
                        labelsize=fontsize, pad=1.5)
        
        plt.subplots_adjust(left=0.03,right=0.97,
                            bottom=0.11,top=0.92,
                            wspace=0,hspace=0)
        plt.savefig(args.plot, dpi=300)
                
