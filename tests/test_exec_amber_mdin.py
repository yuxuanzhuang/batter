from batter.exec.amber.mdin import (
    AmberMdin,
    apply_disang,
    apply_minimization,
    apply_npt,
    apply_restraints,
    apply_ti,
    apply_wt_end,
)


def test_amber_mdin_defaults_and_minimization(tmp_path):
    mdin = AmberMdin()
    default = mdin.to_string()
    assert "&cntrl" in default
    assert "imin = 0" in default

    apply_minimization(mdin, steps=2000)
    minimised = mdin.to_string()
    assert "imin = 1" in minimised
    assert "maxcyc = 2000" in minimised


def test_amber_mdin_ti_and_outputs(tmp_path):
    mdin = AmberMdin()
    apply_ti(
        mdin,
        lbd_val=0.5,
        timask1=":1",
        timask2=":2",
        scmask1=":1",
        scmask2=":2",
        crgmask=":1",
    )
    apply_restraints(mdin, mask=":LIG", weight=25.0)
    apply_npt(mdin, temp=298.0, steps=5000, dt=0.002)
    apply_disang(mdin, filename="restraints.dat")
    apply_wt_end(mdin)

    text = mdin.to_string()
    assert "clambda = 0.5" in text
    assert "restraintmask = ':LIG'" in text
    assert "DISANG=restraints.dat" in text
    assert "&wt type='END'" in text

    out_file = tmp_path / "mdin.in"
    mdin.save(out_file)
    assert out_file.exists()
