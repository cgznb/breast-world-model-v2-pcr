import pytest
import torch
from symm_world.flow import make_path,estimate_endpoints,integrate,local_teacher_map
from symm_world.losses import distribution_mmd,energy_score,enhancement_differences


def observations():
    return torch.randn(2,24,2,3,4),torch.randn(2,24,2,3,4)

@pytest.mark.parametrize("t",[0.,.3,1.])
def test_analytic_endpoints(t):
    a,b=observations();p=make_path(a,b,torch.full((2,),t))
    ah,bh=estimate_endpoints(p.joint,p.velocity,p.tau)
    torch.testing.assert_close(ah,a);torch.testing.assert_close(bh,b)

def test_analytic_velocity_derivative():
    a,b=observations();ex=torch.randn_like(a);ey=torch.randn_like(a)
    tau=torch.tensor([.3,.6]);p=make_path(a,b,tau,ex,ey);q=make_path(a,b,tau+1e-3,ex,ey)
    torch.testing.assert_close((q.joint-p.joint)/1e-3,p.velocity,atol=5e-4,rtol=5e-4)

@pytest.mark.parametrize("method",["euler","heun"])
def test_integrator_direction_and_gradient(method):
    x=torch.randn(1,48,2,2,2,requires_grad=True);s=torch.tensor(.7,requires_grad=True)
    velocity=lambda z,t,c:torch.ones_like(z)*s
    y=integrate(velocity,x,None,steps=3,method=method)
    recovered=integrate(velocity,y,None,steps=3,method=method,direction=-1)
    torch.testing.assert_close(y,x+s);torch.testing.assert_close(recovered,x)
    y.sum().backward();assert s.grad>0 and x.grad.abs().sum()>0

def test_local_map_not_constant_velocity_assumption():
    x=torch.ones(1,48,1,1,1);t=torch.tensor([.2])
    mapped,dt=local_teacher_map(lambda z,t,c:z,x,t,None,.1)
    torch.testing.assert_close(mapped,x*(1+.1+.005))
    assert not torch.equal(mapped,x+dt[:,None,None,None,None]*x)

def test_endpoint_l2_is_weighted_velocity_l2():
    a,b=observations();p=make_path(a,b);pred=p.velocity+torch.randn_like(p.velocity)
    ah,bh=estimate_endpoints(p.joint,pred,p.tau)
    t=p.tau[:,None,None,None,None]
    torch.testing.assert_close((bh-b).square(),((1-t)*(pred[:,:24]-p.velocity[:,:24])).square())
    torch.testing.assert_close((ah-a).square(),(t*(pred[:,24:]-p.velocity[:,24:])).square())

def test_mmd_permutation_and_no_patient_mixing():
    x=torch.randn(3,4,7);y=torch.randn(3,4,7)
    torch.testing.assert_close(distribution_mmd(x,y),distribution_mmd(x[:,[2,0,3,1]],y[:,[1,3,0,2]]))
    torch.testing.assert_close(distribution_mmd(x,x),torch.tensor(0.),atol=1e-6,rtol=0)
    z=x.clone();z[0]+=100
    assert distribution_mmd(x,z)>0
    mean=torch.stack([distribution_mmd(x[i:i+1],y[i:i+1]) for i in range(3)]).mean()
    torch.testing.assert_close(mean,distribution_mmd(x,y))

def test_distribution_gradients_and_k_validation():
    x=torch.randn(2,3,8,requires_grad=True);y=torch.randn(2,3,8)
    loss=distribution_mmd(x,y)+energy_score(x,y[:,0]);loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    with pytest.raises(ValueError):distribution_mmd(x[:,:1],y[:,:1])

def test_image_differences_are_not_latent_physiology():
    x=torch.cat([torch.ones(1,1,2,2,2)*k for k in (1,3,4)],1)
    expected=torch.tensor([2.,3.,1.]);torch.testing.assert_close(enhancement_differences(x)[0,:,0,0,0],expected)
    with pytest.raises(ValueError):enhancement_differences(torch.ones(1,24,2,2,2))

@pytest.mark.parametrize("tau",[torch.tensor([-0.1,0.2]),torch.tensor([.1,1.1]),torch.tensor([float('nan'),.1])])
def test_invalid_flow_time(tau):
    a,b=observations()
    with pytest.raises(ValueError):make_path(a,b,tau)
