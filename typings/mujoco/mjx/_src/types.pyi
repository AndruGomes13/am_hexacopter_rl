"""
Base types used in MJX.
"""
from __future__ import annotations
import dataclasses as dataclasses
import enum as enum
import jax as jax
import mujoco as mujoco
from mujoco.mjx._src.dataclasses import PyTreeNode
import numpy as np
import numpy
import typing
__all__ = ['BiasType', 'CamLightType', 'ConeType', 'ConstraintType', 'Contact', 'ConvexMesh', 'Data', 'DisableBit', 'DynType', 'EqType', 'GainType', 'GeomType', 'IntegratorType', 'JacobianType', 'JointType', 'Model', 'ObjType', 'Option', 'PyTreeNode', 'SensorType', 'SolverType', 'Statistic', 'TrnType', 'WrapType', 'dataclasses', 'enum', 'jax', 'mujoco', 'np']
class BiasType(enum.IntEnum):
    """
    Type of actuator bias.
    
      Members:
        NONE: no bias
        AFFINE: const + kp*length + kv*velocity
        MUSCLE: muscle passive force computed by muscle_bias
      
    """
    AFFINE: typing.ClassVar[BiasType]  # value = <BiasType.AFFINE: 1>
    MUSCLE: typing.ClassVar[BiasType]  # value = <BiasType.MUSCLE: 2>
    NONE: typing.ClassVar[BiasType]  # value = <BiasType.NONE: 0>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class CamLightType(enum.IntEnum):
    """
    Type of camera light.
    
      Members:
        FIXED: pos and rot fixed in body
        TRACK: pos tracks body, rot fixed in global
        TRACKCOM: pos tracks subtree com, rot fixed in body
        TARGETBODY: pos fixed in body, rot tracks target body
        TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
      
    """
    FIXED: typing.ClassVar[CamLightType]  # value = <CamLightType.FIXED: 0>
    TARGETBODY: typing.ClassVar[CamLightType]  # value = <CamLightType.TARGETBODY: 3>
    TARGETBODYCOM: typing.ClassVar[CamLightType]  # value = <CamLightType.TARGETBODYCOM: 4>
    TRACK: typing.ClassVar[CamLightType]  # value = <CamLightType.TRACK: 1>
    TRACKCOM: typing.ClassVar[CamLightType]  # value = <CamLightType.TRACKCOM: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class ConeType(enum.IntEnum):
    """
    Type of friction cone.
    
      Members:
        PYRAMIDAL: pyramidal
        ELLIPTIC: elliptic
      
    """
    ELLIPTIC: typing.ClassVar[ConeType]  # value = <ConeType.ELLIPTIC: 1>
    PYRAMIDAL: typing.ClassVar[ConeType]  # value = <ConeType.PYRAMIDAL: 0>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class ConstraintType(enum.IntEnum):
    """
    Type of constraint.
    
      Members:
        EQUALITY: equality constraint
        LIMIT_JOINT: joint limit
        LIMIT_TENDON: tendon limit
        CONTACT_FRICTIONLESS: frictionless contact
        CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
      
    """
    CONTACT_ELLIPTIC: typing.ClassVar[ConstraintType]  # value = <ConstraintType.CONTACT_ELLIPTIC: 7>
    CONTACT_FRICTIONLESS: typing.ClassVar[ConstraintType]  # value = <ConstraintType.CONTACT_FRICTIONLESS: 5>
    CONTACT_PYRAMIDAL: typing.ClassVar[ConstraintType]  # value = <ConstraintType.CONTACT_PYRAMIDAL: 6>
    EQUALITY: typing.ClassVar[ConstraintType]  # value = <ConstraintType.EQUALITY: 0>
    FRICTION_DOF: typing.ClassVar[ConstraintType]  # value = <ConstraintType.FRICTION_DOF: 1>
    FRICTION_TENDON: typing.ClassVar[ConstraintType]  # value = <ConstraintType.FRICTION_TENDON: 2>
    LIMIT_JOINT: typing.ClassVar[ConstraintType]  # value = <ConstraintType.LIMIT_JOINT: 3>
    LIMIT_TENDON: typing.ClassVar[ConstraintType]  # value = <ConstraintType.LIMIT_TENDON: 4>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class Contact(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Result of collision detection functions.
    
      Attributes:
        dist: distance between nearest points; neg: penetration
        pos: position of contact point: midpoint between geoms            (3,)
        frame: normal is in [0-2]                                         (9,)
        includemargin: include if dist<includemargin=margin-gap           (1,)
        friction: tangent1, 2, spin, roll1, 2                             (5,)
        solref: constraint solver reference, normal direction             (mjNREF,)
        solreffriction: constraint solver reference, friction directions  (mjNREF,)
        solimp: constraint solver impedance                               (mjNIMP,)
        dim: contact space dimensionality: 1, 3, 4, or 6
        geom1: id of geom 1; deprecated, use geom[0]
        geom2: id of geom 2; deprecated, use geom[1]
        geom: geom ids                                                    (2,)
        efc_address: address in efc; -1: not included
      
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'dist': Field(name='dist',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pos': Field(name='pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'frame': Field(name='frame',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'includemargin': Field(name='includemargin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'friction': Field(name='friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solref': Field(name='solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solreffriction': Field(name='solreffriction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solimp': Field(name='solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dim': Field(name='dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom1': Field(name='geom1',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom2': Field(name='geom2',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom': Field(name='geom',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_address': Field(name='efc_address',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = ('dist', 'pos', 'frame', 'includemargin', 'friction', 'solref', 'solreffriction', 'solimp', 'dim', 'geom1', 'geom2', 'geom', 'efc_address')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, dist: jax.Array, pos: jax.Array, frame: jax.Array, includemargin: jax.Array, friction: jax.Array, solref: jax.Array, solreffriction: jax.Array, solimp: jax.Array, dim: numpy.ndarray, geom1: jax.Array, geom2: jax.Array, geom: jax.Array, efc_address: numpy.ndarray) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class ConvexMesh(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Geom properties for convex meshes.
    
      Members:
        vert: vertices of the convex mesh
        face: faces of the convex mesh
        face_normal: normal vectors for the faces
        edge: edge indexes for all edges in the convex mesh
        edge_face_normal: indexes for face normals adjacent to edges in `edge`
      
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'vert': Field(name='vert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'face': Field(name='face',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'face_normal': Field(name='face_normal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'edge': Field(name='edge',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'edge_face_normal': Field(name='edge_face_normal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = ('vert', 'face', 'face_normal', 'edge', 'edge_face_normal')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, vert: jax.Array, face: jax.Array, face_normal: jax.Array, edge: jax.Array, edge_face_normal: jax.Array) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class Data(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    \\Dynamic state that updates each step.
    
      Attributes:
        ne: number of equality constraints
        nf: number of friction constraints
        nl: number of limit constraints
        nefc: number of constraints
        ncon: number of contacts
        solver_niter: number of solver iterations
        time: simulation time
        qpos: position                                              (nq,)
        qvel: velocity                                              (nv,)
        act: actuator activation                                    (na,)
        qacc_warmstart: acceleration used for warmstart             (nv,)
        ctrl: control                                               (nu,)
        qfrc_applied: applied generalized force                     (nv,)
        xfrc_applied: applied Cartesian force/torque                (nbody, 6)
        eq_active: enable/disable constraints                       (neq,)
        mocap_pos: positions of mocap bodies                        (nmocap x 3)
        mocap_quat: orientations of mocap bodies                    (nmocap x 4)
        qacc: acceleration                                          (nv,)
        act_dot: time-derivative of actuator activation             (na,)
        userdata: user data, not touched by engine                  (nuserdata,)
        sensordata: sensor data array                               (nsensordata,)
        xpos:  Cartesian position of body frame                     (nbody, 3)
        xquat: Cartesian orientation of body frame                  (nbody, 4)
        xmat:  Cartesian orientation of body frame                  (nbody, 3, 3)
        xipos: Cartesian position of body com                       (nbody, 3)
        ximat: Cartesian orientation of body inertia                (nbody, 3, 3)
        xanchor: Cartesian position of joint anchor                 (njnt, 3)
        xaxis: Cartesian joint axis                                 (njnt, 3)
        geom_xpos: Cartesian geom position                          (ngeom, 3)
        geom_xmat: Cartesian geom orientation                       (ngeom, 3, 3)
        site_xpos: Cartesian site position                          (nsite, 3)
        site_xmat: Cartesian site orientation                       (nsite, 3, 3)
        cam_xpos: Cartesian camera position                         (ncam, 3)
        cam_xmat: Cartesian camera orientation                      (ncam, 3, 3)
        light_xpos: Cartesian light position                        (nlight, 3)
        light_xdir: Cartesian light direction                       (nlight, 3)
        subtree_com: center of mass of each subtree                 (nbody, 3)
        cdof: com-based motion axis of each dof                     (nv, 6)
        cinert: com-based body inertia and mass                     (nbody, 10)
        flexvert_xpos: Cartesian flex vertex positions              (nflexvert, 3)
        flexelem_aabb: flex element bounding boxes (center, size)   (nflexelem, 6)
        flexedge_J_rownnz: number of non-zeros in Jacobian row      (nflexedge,)
        flexedge_J_rowadr: row start address in colind array        (nflexedge,)
        flexedge_J_colind: column indices in sparse Jacobian        (nflexedge, nv)
        flexedge_J: flex edge Jacobian                              (nflexedge, nv)
        flexedge_length: flex edge lengths                          (nflexedge,)
        ten_wrapadr: start address of tendon's path                 (ntendon,)
        ten_wrapnum: number of wrap points in path                  (ntendon,)
        ten_J_rownnz: number of non-zeros in Jacobian row           (ntendon,)
        ten_J_rowadr: row start address in colind array             (ntendon,)
        ten_J_colind: column indices in sparse Jacobian             (ntendon, nv)
        ten_J: tendon Jacobian                                      (ntendon, nv)
        ten_length: tendon lengths                                  (ntendon,)
        wrap_obj: geom id; -1: site; -2: pulley                     (nwrap*2,)
        wrap_xpos: Cartesian 3D points in all path                  (nwrap*2, 3)
        actuator_length: actuator lengths                           (nu,)
        moment_rownnz: number of non-zeros in actuator_moment row   (nu,)
        moment_rowadr: row start address in colind array            (nu,)
        moment_colind: column indices in sparse Jacobian            (nJmom,)
        actuator_moment: actuator moments                           (nJmom,)
        crb: com-based composite inertia and mass                   (nbody, 10)
        qM: total inertia                                if sparse: (nM,)
                                                         if dense:  (nv, nv)
        qLD: L'*D*L (or Cholesky) factorization of M.    if sparse: (nM,)
                                                         if dense:  (nv, nv)
        qLDiagInv: 1/diag(D)                             if sparse: (nv,)
                                                         if dense:  (0,)
        bvh_aabb_dyn: global bounding box (center, size)            (nbvhdynamic, 6)
        bvh_active: volume has been added to collisions             (nbvh,)
        flexedge_velocity: flex edge velocities                     (nflexedge,)
        ten_velocity: tendon velocities                             (ntendon,)
        actuator_velocity: actuator velocities                      (nu,)
        cvel: com-based velocity [3D rot; 3D tran]                  (nbody, 6)
        cdof_dot: time-derivative of cdof                           (nv, 6)
        qfrc_bias: C(qpos,qvel)                                     (nv,)
        qfrc_spring: passive spring force                           (nv,)
        qfrc_damper: passive damper force                           (nv,)
        qfrc_gravcomp: passive gravity compensation force           (nv,)
        qfrc_fluid: passive fluid force                             (nv,)
        qfrc_passive: total passive force                           (nv,)
        subtree_linvel: linear velocity of subtree com              (nbody, 3)
        subtree_angmom: angular momentum about subtree com          (nbody, 3)
        qH: L'*D*L factorization of modified M                      (nM,)
        qHDiagInv: 1/diag(D) of modified M                          (nv,)
        B_rownnz: body-dof: non-zeros in each row                   (nbody,)
        B_rowadr: body-dof: address of each row in B_colind         (nbody,)
        B_colind: body-dof: column indices of non-zeros             (nB,)
        C_rownnz: reduced dof-dof: non-zeros in each row            (nv,)
        C_rowadr: reduced dof-dof: address of each row in C_colind  (nv,)
        C_colind: reduced dof-dof: column indices of non-zeros      (nC,)
        mapM2C: index mapping from M to C                           (nC,)
        D_rownnz: dof-dof: non-zeros in each row                    (nv,)
        D_rowadr: dof-dof: address of each row in D_colind          (nv,)
        D_diag: dof-dof: index of diagonal element                  (nv,)
        D_colind: dof-dof: column indices of non-zeros              (nD,)
        mapM2D: index mapping from M to D                           (nD,)
        mapD2M: index mapping from D to M                           (nM,)
        qDeriv: d (passive + actuator - bias) / d qvel              (nD,)
        qLU: sparse LU of (qM - dt*qDeriv)                          (nD,)
        actuator_force: actuator force in actuation space           (nu,)
        qfrc_actuator: actuator force                               (nv,)
        qfrc_smooth: net unconstrained force                        (nv,)
        qacc_smooth: unconstrained acceleration                     (nv,)
        qfrc_constraint: constraint force                           (nv,)
        qfrc_inverse: net external force; should equal:             (nv,)
                      qfrc_applied + J'*xfrc_applied + qfrc_actuator
        cacc: com-based acceleration                                (nbody, 6)
        cfrc_int: com-based interaction force with parent           (nbody, 6)
        cfrc_ext: com-based external force on body                  (nbody, 6)
        contact: all detected contacts                              (ncon,)
        efc_type: constraint type                                   (nefc,)
        efc_J: constraint Jacobian                                  (nefc, nv)
        efc_pos: constraint position (equality, contact)            (nefc,)
        efc_margin: inclusion margin (contact)                      (nefc,)
        efc_frictionloss: frictionloss (friction)                   (nefc,)
        efc_D: constraint mass                                      (nefc,)
        efc_aref: reference pseudo-acceleration                     (nefc,)
        efc_force: constraint force in constraint space             (nefc,)
        _qM_sparse: qM in sparse representation                     (nM,)
        _qLD_sparse: qLD in sparse representation                   (nC,)
        _qLDiagInv_sparse: qLDiagInv in sparse representation       (nv,)
      
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'ne': Field(name='ne',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nf': Field(name='nf',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nl': Field(name='nl',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nefc': Field(name='nefc',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ncon': Field(name='ncon',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solver_niter': Field(name='solver_niter',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'time': Field(name='time',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qpos': Field(name='qpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qvel': Field(name='qvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'act': Field(name='act',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qacc_warmstart': Field(name='qacc_warmstart',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ctrl': Field(name='ctrl',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_applied': Field(name='qfrc_applied',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xfrc_applied': Field(name='xfrc_applied',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_active': Field(name='eq_active',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mocap_pos': Field(name='mocap_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mocap_quat': Field(name='mocap_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qacc': Field(name='qacc',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'act_dot': Field(name='act_dot',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'userdata': Field(name='userdata',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensordata': Field(name='sensordata',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xpos': Field(name='xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xquat': Field(name='xquat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xmat': Field(name='xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xipos': Field(name='xipos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ximat': Field(name='ximat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xanchor': Field(name='xanchor',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xaxis': Field(name='xaxis',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_xpos': Field(name='geom_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_xmat': Field(name='geom_xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_xpos': Field(name='site_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_xmat': Field(name='site_xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_xpos': Field(name='cam_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_xmat': Field(name='cam_xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_xpos': Field(name='light_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'light_xdir': Field(name='light_xdir',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'subtree_com': Field(name='subtree_com',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cdof': Field(name='cdof',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cinert': Field(name='cinert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexvert_xpos': Field(name='flexvert_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexelem_aabb': Field(name='flexelem_aabb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_J_rownnz': Field(name='flexedge_J_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_J_rowadr': Field(name='flexedge_J_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_J_colind': Field(name='flexedge_J_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_J': Field(name='flexedge_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_length': Field(name='flexedge_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ten_wrapadr': Field(name='ten_wrapadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_wrapnum': Field(name='ten_wrapnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_J_rownnz': Field(name='ten_J_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ten_J_rowadr': Field(name='ten_J_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ten_J_colind': Field(name='ten_J_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ten_J': Field(name='ten_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_length': Field(name='ten_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_obj': Field(name='wrap_obj',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_xpos': Field(name='wrap_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_length': Field(name='actuator_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'moment_rownnz': Field(name='moment_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'moment_rowadr': Field(name='moment_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'moment_colind': Field(name='moment_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'actuator_moment': Field(name='actuator_moment',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'crb': Field(name='crb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qM': Field(name='qM',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLD': Field(name='qLD',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLDiagInv': Field(name='qLDiagInv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'bvh_aabb_dyn': Field(name='bvh_aabb_dyn',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'bvh_active': Field(name='bvh_active',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_velocity': Field(name='flexedge_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ten_velocity': Field(name='ten_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_velocity': Field(name='actuator_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cvel': Field(name='cvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cdof_dot': Field(name='cdof_dot',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_bias': Field(name='qfrc_bias',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_spring': Field(name='qfrc_spring',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'qfrc_damper': Field(name='qfrc_damper',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'qfrc_gravcomp': Field(name='qfrc_gravcomp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_fluid': Field(name='qfrc_fluid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_passive': Field(name='qfrc_passive',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_linvel': Field(name='subtree_linvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_angmom': Field(name='subtree_angmom',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qH': Field(name='qH',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'qHDiagInv': Field(name='qHDiagInv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'B_rownnz': Field(name='B_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'B_rowadr': Field(name='B_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'B_colind': Field(name='B_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'C_rownnz': Field(name='C_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'C_rowadr': Field(name='C_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'C_colind': Field(name='C_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'mapM2C': Field(name='mapM2C',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'D_rownnz': Field(name='D_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'D_rowadr': Field(name='D_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'D_diag': Field(name='D_diag',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'D_colind': Field(name='D_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'mapM2D': Field(name='mapM2D',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'mapD2M': Field(name='mapD2M',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'qDeriv': Field(name='qDeriv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'qLU': Field(name='qLU',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'qfrc_actuator': Field(name='qfrc_actuator',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_force': Field(name='actuator_force',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_smooth': Field(name='qfrc_smooth',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qacc_smooth': Field(name='qacc_smooth',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_constraint': Field(name='qfrc_constraint',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_inverse': Field(name='qfrc_inverse',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cacc': Field(name='cacc',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cfrc_int': Field(name='cfrc_int',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cfrc_ext': Field(name='cfrc_ext',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'contact': Field(name='contact',type=<class 'mujoco.mjx._src.types.Contact'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_type': Field(name='efc_type',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_J': Field(name='efc_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_pos': Field(name='efc_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_margin': Field(name='efc_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_frictionloss': Field(name='efc_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_D': Field(name='efc_D',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_aref': Field(name='efc_aref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_force': Field(name='efc_force',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), '_qM_sparse': Field(name='_qM_sparse',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), '_qLD_sparse': Field(name='_qLD_sparse',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), '_qLDiagInv_sparse': Field(name='_qLDiagInv_sparse',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = ('ne', 'nf', 'nl', 'nefc', 'ncon', 'solver_niter', 'time', 'qpos', 'qvel', 'act', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'qacc', 'act_dot', 'userdata', 'sensordata', 'xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'xanchor', 'xaxis', 'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir', 'subtree_com', 'cdof', 'cinert', 'flexvert_xpos', 'flexelem_aabb', 'flexedge_J_rownnz', 'flexedge_J_rowadr', 'flexedge_J_colind', 'flexedge_J', 'flexedge_length', 'ten_wrapadr', 'ten_wrapnum', 'ten_J_rownnz', 'ten_J_rowadr', 'ten_J_colind', 'ten_J', 'ten_length', 'wrap_obj', 'wrap_xpos', 'actuator_length', 'moment_rownnz', 'moment_rowadr', 'moment_colind', 'actuator_moment', 'crb', 'qM', 'qLD', 'qLDiagInv', 'bvh_aabb_dyn', 'bvh_active', 'flexedge_velocity', 'ten_velocity', 'actuator_velocity', 'cvel', 'cdof_dot', 'qfrc_bias', 'qfrc_spring', 'qfrc_damper', 'qfrc_gravcomp', 'qfrc_fluid', 'qfrc_passive', 'subtree_linvel', 'subtree_angmom', 'qH', 'qHDiagInv', 'B_rownnz', 'B_rowadr', 'B_colind', 'C_rownnz', 'C_rowadr', 'C_colind', 'mapM2C', 'D_rownnz', 'D_rowadr', 'D_diag', 'D_colind', 'mapM2D', 'mapD2M', 'qDeriv', 'qLU', 'qfrc_actuator', 'actuator_force', 'qfrc_smooth', 'qacc_smooth', 'qfrc_constraint', 'qfrc_inverse', 'cacc', 'cfrc_int', 'cfrc_ext', 'contact', 'efc_type', 'efc_J', 'efc_pos', 'efc_margin', 'efc_frictionloss', 'efc_D', 'efc_aref', 'efc_force', '_qM_sparse', '_qLD_sparse', '_qLDiagInv_sparse')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, ne: int, nf: int, nl: int, nefc: int, ncon: int, solver_niter: jax.Array, time: jax.Array, qpos: jax.Array, qvel: jax.Array, act: jax.Array, qacc_warmstart: jax.Array, ctrl: jax.Array, qfrc_applied: jax.Array, xfrc_applied: jax.Array, eq_active: jax.Array, mocap_pos: jax.Array, mocap_quat: jax.Array, qacc: jax.Array, act_dot: jax.Array, userdata: jax.Array, sensordata: jax.Array, xpos: jax.Array, xquat: jax.Array, xmat: jax.Array, xipos: jax.Array, ximat: jax.Array, xanchor: jax.Array, xaxis: jax.Array, geom_xpos: jax.Array, geom_xmat: jax.Array, site_xpos: jax.Array, site_xmat: jax.Array, cam_xpos: jax.Array, cam_xmat: jax.Array, light_xpos: jax.Array, light_xdir: jax.Array, subtree_com: jax.Array, cdof: jax.Array, cinert: jax.Array, flexvert_xpos: jax.Array, flexelem_aabb: jax.Array, flexedge_J_rownnz: jax.Array, flexedge_J_rowadr: jax.Array, flexedge_J_colind: jax.Array, flexedge_J: jax.Array, flexedge_length: jax.Array, ten_wrapadr: jax.Array, ten_wrapnum: jax.Array, ten_J_rownnz: jax.Array, ten_J_rowadr: jax.Array, ten_J_colind: jax.Array, ten_J: jax.Array, ten_length: jax.Array, wrap_obj: jax.Array, wrap_xpos: jax.Array, actuator_length: jax.Array, moment_rownnz: jax.Array, moment_rowadr: jax.Array, moment_colind: jax.Array, actuator_moment: jax.Array, crb: jax.Array, qM: jax.Array, qLD: jax.Array, qLDiagInv: jax.Array, bvh_aabb_dyn: jax.Array, bvh_active: jax.Array, flexedge_velocity: jax.Array, ten_velocity: jax.Array, actuator_velocity: jax.Array, cvel: jax.Array, cdof_dot: jax.Array, qfrc_bias: jax.Array, qfrc_spring: jax.Array, qfrc_damper: jax.Array, qfrc_gravcomp: jax.Array, qfrc_fluid: jax.Array, qfrc_passive: jax.Array, subtree_linvel: jax.Array, subtree_angmom: jax.Array, qH: jax.Array, qHDiagInv: jax.Array, B_rownnz: jax.Array, B_rowadr: jax.Array, B_colind: jax.Array, C_rownnz: jax.Array, C_rowadr: jax.Array, C_colind: jax.Array, mapM2C: jax.Array, D_rownnz: jax.Array, D_rowadr: jax.Array, D_diag: jax.Array, D_colind: jax.Array, mapM2D: jax.Array, mapD2M: jax.Array, qDeriv: jax.Array, qLU: jax.Array, qfrc_actuator: jax.Array, actuator_force: jax.Array, qfrc_smooth: jax.Array, qacc_smooth: jax.Array, qfrc_constraint: jax.Array, qfrc_inverse: jax.Array, cacc: jax.Array, cfrc_int: jax.Array, cfrc_ext: jax.Array, contact: Contact, efc_type: jax.Array, efc_J: jax.Array, efc_pos: jax.Array, efc_margin: jax.Array, efc_frictionloss: jax.Array, efc_D: jax.Array, efc_aref: jax.Array, efc_force: jax.Array, _qM_sparse: jax.Array, _qLD_sparse: jax.Array, _qLDiagInv_sparse: jax.Array) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
    def bind(self, model: Model, obj: typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin, collections.abc.Iterable[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]]]) -> mujoco.mjx._src.support.BindData:
        """
        Bind a Mujoco spec to an MJX Data.
        """
class DisableBit(enum.IntFlag):
    """
    Disable default feature bitflags.
    
      Members:
        CONSTRAINT:   entire constraint solver
        EQUALITY:     equality constraints
        FRICTIONLOSS: joint and tendon frictionloss constraints
        LIMIT:        joint and tendon limit constraints
        CONTACT:      contact constraints
        PASSIVE:      passive forces
        GRAVITY:      gravitational forces
        CLAMPCTRL:    clamp control to specified range
        WARMSTART:    warmstart constraint solver
        ACTUATION:    apply actuation forces
        REFSAFE:      integrator safety: make ref[0]>=2*timestep
        SENSOR:       sensors
      
    """
    ACTUATION: typing.ClassVar[DisableBit]  # value = <DisableBit.ACTUATION: 1024>
    CLAMPCTRL: typing.ClassVar[DisableBit]  # value = <DisableBit.CLAMPCTRL: 128>
    CONSTRAINT: typing.ClassVar[DisableBit]  # value = <DisableBit.CONSTRAINT: 1>
    CONTACT: typing.ClassVar[DisableBit]  # value = <DisableBit.CONTACT: 16>
    EQUALITY: typing.ClassVar[DisableBit]  # value = <DisableBit.EQUALITY: 2>
    EULERDAMP: typing.ClassVar[DisableBit]  # value = <DisableBit.EULERDAMP: 16384>
    FILTERPARENT: typing.ClassVar[DisableBit]  # value = <DisableBit.FILTERPARENT: 512>
    FRICTIONLOSS: typing.ClassVar[DisableBit]  # value = <DisableBit.FRICTIONLOSS: 4>
    GRAVITY: typing.ClassVar[DisableBit]  # value = <DisableBit.GRAVITY: 64>
    LIMIT: typing.ClassVar[DisableBit]  # value = <DisableBit.LIMIT: 8>
    PASSIVE: typing.ClassVar[DisableBit]  # value = <DisableBit.PASSIVE: 32>
    REFSAFE: typing.ClassVar[DisableBit]  # value = <DisableBit.REFSAFE: 2048>
    SENSOR: typing.ClassVar[DisableBit]  # value = <DisableBit.SENSOR: 4096>
    WARMSTART: typing.ClassVar[DisableBit]  # value = <DisableBit.WARMSTART: 256>
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        ...
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
class DynType(enum.IntEnum):
    """
    Type of actuator dynamics.
    
      Members:
        NONE: no internal dynamics; ctrl specifies force
        INTEGRATOR: integrator: da/dt = u
        FILTER: linear filter: da/dt = (u-a) / tau
        FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
        MUSCLE: piece-wise linear filter with two time constants
      
    """
    FILTER: typing.ClassVar[DynType]  # value = <DynType.FILTER: 2>
    FILTEREXACT: typing.ClassVar[DynType]  # value = <DynType.FILTEREXACT: 3>
    INTEGRATOR: typing.ClassVar[DynType]  # value = <DynType.INTEGRATOR: 1>
    MUSCLE: typing.ClassVar[DynType]  # value = <DynType.MUSCLE: 4>
    NONE: typing.ClassVar[DynType]  # value = <DynType.NONE: 0>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class EqType(enum.IntEnum):
    """
    Type of equality constraint.
    
      Members:
        CONNECT: connect two bodies at a point (ball joint)
        WELD: fix relative position and orientation of two bodies
        JOINT: couple the values of two scalar joints with cubic
        TENDON: couple the lengths of two tendons with cubic
      
    """
    CONNECT: typing.ClassVar[EqType]  # value = <EqType.CONNECT: 0>
    JOINT: typing.ClassVar[EqType]  # value = <EqType.JOINT: 2>
    TENDON: typing.ClassVar[EqType]  # value = <EqType.TENDON: 3>
    WELD: typing.ClassVar[EqType]  # value = <EqType.WELD: 1>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class GainType(enum.IntEnum):
    """
    Type of actuator gain.
    
      Members:
        FIXED: fixed gain
        AFFINE: const + kp*length + kv*velocity
        MUSCLE: muscle FLV curve computed by muscle_gain
      
    """
    AFFINE: typing.ClassVar[GainType]  # value = <GainType.AFFINE: 1>
    FIXED: typing.ClassVar[GainType]  # value = <GainType.FIXED: 0>
    MUSCLE: typing.ClassVar[GainType]  # value = <GainType.MUSCLE: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class GeomType(enum.IntEnum):
    """
    Type of geometry.
    
      Members:
        PLANE: plane
        HFIELD: height field
        SPHERE: sphere
        CAPSULE: capsule
        ELLIPSOID: ellipsoid
        CYLINDER: cylinder
        BOX: box
        MESH: mesh
        SDF: signed distance field
      
    """
    BOX: typing.ClassVar[GeomType]  # value = <GeomType.BOX: 6>
    CAPSULE: typing.ClassVar[GeomType]  # value = <GeomType.CAPSULE: 3>
    CYLINDER: typing.ClassVar[GeomType]  # value = <GeomType.CYLINDER: 5>
    ELLIPSOID: typing.ClassVar[GeomType]  # value = <GeomType.ELLIPSOID: 4>
    HFIELD: typing.ClassVar[GeomType]  # value = <GeomType.HFIELD: 1>
    MESH: typing.ClassVar[GeomType]  # value = <GeomType.MESH: 7>
    PLANE: typing.ClassVar[GeomType]  # value = <GeomType.PLANE: 0>
    SPHERE: typing.ClassVar[GeomType]  # value = <GeomType.SPHERE: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class IntegratorType(enum.IntEnum):
    """
    Integrator mode.
    
      Members:
        EULER: semi-implicit Euler
        RK4: 4th-order Runge Kutta
        IMPLICITFAST: implicit in velocity, no rne derivative
      
    """
    EULER: typing.ClassVar[IntegratorType]  # value = <IntegratorType.EULER: 0>
    IMPLICITFAST: typing.ClassVar[IntegratorType]  # value = <IntegratorType.IMPLICITFAST: 3>
    RK4: typing.ClassVar[IntegratorType]  # value = <IntegratorType.RK4: 1>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class JacobianType(enum.IntEnum):
    """
    Type of constraint Jacobian.
    
      Members:
        DENSE: dense
        SPARSE: sparse
        AUTO: sparse if nv>60 and device is TPU, dense otherwise
      
    """
    AUTO: typing.ClassVar[JacobianType]  # value = <JacobianType.AUTO: 2>
    DENSE: typing.ClassVar[JacobianType]  # value = <JacobianType.DENSE: 0>
    SPARSE: typing.ClassVar[JacobianType]  # value = <JacobianType.SPARSE: 1>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class JointType(enum.IntEnum):
    """
    Type of degree of freedom.
    
      Members:
        FREE:  global position and orientation (quat)       (7,)
        BALL:  orientation (quat) relative to parent        (4,)
        SLIDE: sliding distance along body-fixed axis       (1,)
        HINGE: rotation angle (rad) around body-fixed axis  (1,)
      
    """
    BALL: typing.ClassVar[JointType]  # value = <JointType.BALL: 1>
    FREE: typing.ClassVar[JointType]  # value = <JointType.FREE: 0>
    HINGE: typing.ClassVar[JointType]  # value = <JointType.HINGE: 3>
    SLIDE: typing.ClassVar[JointType]  # value = <JointType.SLIDE: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class Model(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Static model of the scene that remains unchanged with each physics step.
    
      Attributes:
        nq: number of generalized coordinates = dim(qpos)
        nv: number of degrees of freedom = dim(qvel)
        nu: number of actuators/controls = dim(ctrl)
        na: number of activation states = dim(act)
        nbody: number of bodies
        nbvh: number of total bounding volumes in all bodies
        nbvhstatic: number of static bounding volumes (aabb stored in mjModel)
        nbvhdynamic: number of dynamic bounding volumes (aabb stored in mjData)
        njnt: number of joints
        ngeom: number of geoms
        nsite: number of sites
        ncam: number of cameras
        nlight: number of lights
        nflex: number of flexes
        nflexvert: number of vertices in all flexes
        nflexedge: number of edges in all flexes
        nflexelem: number of elements in all flexes
        nflexelemdata: number of element vertex ids in all flexes
        nflexshelldata: number of shell fragment vertex ids in all flexes
        nflexevpair: number of element-vertex pairs in all flexes
        nflextexcoord: number of vertices with texture coordinates
        nmesh: number of meshes
        nmeshvert: number of vertices in all meshes
        nmeshnormal: number of normals in all meshes
        nmeshtexcoord: number of texcoords in all meshes
        nmeshface: number of triangular faces in all meshes
        nmeshgraph: number of ints in mesh auxiliary data
        nhfield: number of heightfields
        nhfielddata: number of data points in all heightfields
        ntex: number of textures
        ntexdata: number of bytes in texture rgb data
        nmat: number of materials
        npair: number of predefined geom pairs
        nexclude: number of excluded geom pairs
        neq: number of equality constraints
        ntendon: number of tendons
        nwrap: number of wrap objects in all tendon paths
        nsensor: number of sensors
        nnumeric: number of numeric custom fields
        ntuple: number of tuple custom fields
        nkey: number of keyframes
        nmocap: number of mocap bodies
        nM: number of non-zeros in sparse inertia matrix
        nD: number of non-zeros in sparse dof-dof matrix
        nB: number of non-zeros in sparse body-dof matrix
        nC: number of non-zeros in sparse reduced dof-dof matrix
        nD: number of non-zeros in sparse dof-dof matrix
        nJmom: number of non-zeros in sparse actuator_moment matrix
        ntree: number of kinematic trees under world body
        ngravcomp: number of bodies with nonzero gravcomp
        nuserdata: size of userdata array
        nsensordata: number of mjtNums in sensor data vector
        narena: number of bytes in the mjData arena (inclusive of stack)
        opt: physics options
        stat: model statistics
        qpos0: qpos values at default pose                        (nq,)
        qpos_spring: reference pose for springs                   (nq,)
        body_parentid: id of body's parent                        (nbody,)
        body_rootid: id of root above body                        (nbody,)
        body_weldid: id of body that this body is welded to       (nbody,)
        body_jntnum: number of joints for this body               (nbody,)
        body_jntadr: start addr of joints; -1: no joints          (nbody,)
        body_dofnum: number of motion degrees of freedom          (nbody,)
        body_dofadr: start addr of dofs; -1: no dofs              (nbody,)
        body_treeid: id of body's kinematic tree; -1: static      (nbody,)
        body_geomnum: number of geoms                             (nbody,)
        body_geomadr: start addr of geoms; -1: no geoms           (nbody,)
        body_simple: 1: diag M; 2: diag M, sliders only           (nbody,)
        body_pos: position offset rel. to parent body             (nbody, 3)
        body_quat: orientation offset rel. to parent body         (nbody, 4)
        body_ipos: local position of center of mass               (nbody, 3)
        body_iquat: local orientation of inertia ellipsoid        (nbody, 4)
        body_mass: mass                                           (nbody,)
        body_subtreemass: mass of subtree starting at this body   (nbody,)
        body_inertia: diagonal inertia in ipos/iquat frame        (nbody, 3)
        body_gravcomp: antigravity force, units of body weight    (nbody,)
        body_margin: MAX over all geom margins                    (nbody,)
        body_contype: OR over all geom contypes                   (nbody,)
        body_conaffinity: OR over all geom conaffinities          (nbody,)
        body_bvhadr: address of bvh root                          (nbody,)
        body_bvhnum: number of bounding volumes                   (nbody,)
        bvh_child: left and right children in tree                (nbvh, 2)
        bvh_nodeid: geom or elem id of node; -1: non-leaf         (nbvh,)
        bvh_aabb: local bounding box (center, size)               (nbvhstatic, 6)
        body_invweight0: mean inv inert in qpos0 (trn, rot)       (nbody, 2)
        jnt_type: type of joint (mjtJoint)                        (njnt,)
        jnt_qposadr: start addr in 'qpos' for joint's data        (njnt,)
        jnt_dofadr: start addr in 'qvel' for joint's data         (njnt,)
        jnt_bodyid: id of joint's body                            (njnt,)
        jnt_group: group for visibility                           (njnt,)
        jnt_limited: does joint have limits                       (njnt,)
        jnt_actfrclimited: does joint have actuator force limits  (njnt,)
        jnt_actgravcomp: is gravcomp force applied via actuators  (njnt,)
        jnt_solref: constraint solver reference: limit            (njnt, mjNREF)
        jnt_solimp: constraint solver impedance: limit            (njnt, mjNIMP)
        jnt_pos: local anchor position                            (njnt, 3)
        jnt_axis: local joint axis                                (njnt, 3)
        jnt_stiffness: stiffness coefficient                      (njnt,)
        jnt_range: joint limits                                   (njnt, 2)
        jnt_actfrcrange: range of total actuator force            (njnt, 2)
        jnt_margin: min distance for limit detection              (njnt,)
        dof_bodyid: id of dof's body                              (nv,)
        dof_jntid: id of dof's joint                              (nv,)
        dof_parentid: id of dof's parent; -1: none                (nv,)
        dof_treeid: id of dof's kinematic tree                    (nv,)
        dof_Madr: dof address in M-diagonal                       (nv,)
        dof_simplenum: number of consecutive simple dofs          (nv,)
        dof_solref: constraint solver reference:frictionloss      (nv, mjNREF)
        dof_solimp: constraint solver impedance:frictionloss      (nv, mjNIMP)
        dof_frictionloss: dof friction loss                       (nv,)
        dof_hasfrictionloss: dof has >0 frictionloss (MJX)        (nv,)
        dof_armature: dof armature inertia/mass                   (nv,)
        dof_damping: damping coefficient                          (nv,)
        dof_invweight0: diag. inverse inertia in qpos0            (nv,)
        dof_M0: diag. inertia in qpos0                            (nv,)
        geom_type: geometric type (mjtGeom)                       (ngeom,)
        geom_contype: geom contact type                           (ngeom,)
        geom_conaffinity: geom contact affinity                   (ngeom,)
        geom_condim: contact dimensionality (1, 3, 4, 6)          (ngeom,)
        geom_bodyid: id of geom's body                            (ngeom,)
        geom_dataid: id of geom's mesh/hfield; -1: none           (ngeom,)
        geom_group: group for visibility                          (ngeom,)
        geom_matid: material id for rendering                     (ngeom,)
        geom_priority: geom contact priority                      (ngeom,)
        geom_solmix: mixing coef for solref/imp in geom pair      (ngeom,)
        geom_solref: constraint solver reference: contact         (ngeom, mjNREF)
        geom_solimp: constraint solver impedance: contact         (ngeom, mjNIMP)
        geom_size: geom-specific size parameters                  (ngeom, 3)
        geom_aabb: bounding box, (center, size)                   (ngeom, 6)
        geom_rbound: radius of bounding sphere                    (ngeom,)
        geom_rbound_hfield: static rbound for hfield grid bounds  (ngeom,)
        geom_pos: local position offset rel. to body              (ngeom, 3)
        geom_quat: local orientation offset rel. to body          (ngeom, 4)
        geom_friction: friction for (slide, spin, roll)           (ngeom, 3)
        geom_margin: include in solver if dist<margin-gap         (ngeom,)
        geom_gap: include in solver if dist<margin-gap            (ngeom,)
        geom_rgba: rgba when material is omitted                  (ngeom, 4)
        site_bodyid: id of site's body                            (nsite,)
        site_pos: local position offset rel. to body              (nsite, 3)
        site_quat: local orientation offset rel. to body          (nsite, 4)
        cam_mode:  camera tracking mode (mjtCamLight)             (ncam,)
        cam_bodyid:  id of camera's body                          (ncam,)
        cam_targetbodyid:  id of targeted body; -1: none          (ncam,)
        cam_pos:  position rel. to body frame                     (ncam, 3)
        cam_quat:  orientation rel. to body frame                 (ncam, 4)
        cam_poscom0:  global position rel. to sub-com in qpos0    (ncam, 3)
        cam_pos0: global position rel. to body in qpos0           (ncam, 3)
        cam_mat0: global orientation in qpos0                     (ncam, 3, 3)
        cam_fovy: y field-of-view                                 (ncam,)
        cam_resolution: resolution: pixels                        (ncam, 2)
        cam_sensorsize: sensor size: length                       (ncam, 2)
        cam_intrinsic: [focal length; principal point]            (ncam, 4)
        light_mode: light tracking mode (mjtCamLight)             (nlight,)
        light_bodyid: id of light's body                          (nlight,)
        light_targetbodyid: id of targeted body; -1: none         (nlight,)
        light_directional: directional light                      (nlight,)
        light_castshadow: does light cast shadows                 (nlight,)
        light_pos: position rel. to body frame                    (nlight, 3)
        light_dir: direction rel. to body frame                   (nlight, 3)
        light_poscom0: global position rel. to sub-com in qpos0   (nlight, 3)
        light_pos0: global position rel. to body in qpos0         (nlight, 3)
        light_dir0: global direction in qpos0                     (nlight, 3)
        light_cutoff: OpenGL cutoff                               (nlight,)
        flex_contype: flex contact type                           (nflex,)
        flex_conaffinity: flex contact affinity                   (nflex,)
        flex_condim: contact dimensionality (1, 3, 4, 6)          (nflex,)
        flex_priority: flex contact priority                      (nflex,)
        flex_solmix: mix coef for solref/imp in contact pair      (nflex,)
        flex_solref: constraint solver reference: contact         (nflex, mjNREF)
        flex_solimp: constraint solver impedance: contact         (nflex, mjNIMP)
        flex_friction: friction for (slide, spin, roll)           (nflex,)
        flex_margin: detect contact if dist<margin                (nflex,)
        flex_gap: include in solver if dist<margin-gap            (nflex,)
        flex_internal: internal flex collision enabled            (nflex,)
        flex_selfcollide: self collision mode (mjtFlexSelf)       (nflex,)
        flex_activelayers: number of active element layers, 3D only  (nflex,)
        flex_dim: 1: lines, 2: triangles, 3: tetrahedra           (nflex,)
        flex_vertadr: first vertex address                        (nflex,)
        flex_vertnum: number of vertices                          (nflex,)
        flex_edgeadr: first edge address                          (nflex,)
        flex_edgenum: number of edges                             (nflex,)
        flex_elemadr: first element address                       (nflex,)
        flex_elemnum: number of elements                          (nflex,)
        flex_elemdataadr: first element vertex id address         (nflex,)
        flex_evpairadr: first evpair address                      (nflex,)
        flex_evpairnum: number of evpairs                         (nflex,)
        flex_vertbodyid: vertex body ids                          (nflex,)
        flex_edge: edge vertex ids (2 per edge)                   (nflexedge, 2)
        flex_elem: element vertex ids (dim+1 per elem)            (nflexelemdata,)
        flex_elemlayer: element distance from surface, 3D only    (nflexelem,)
        flex_evpair: (element, vertex) collision pairs            (nflexevpair, 2)
        flex_vert: vertex positions in local body frames          (nflexvert, 3)
        flexedge_length0: edge lengths in qpos0                   (nflexedge,)
        flexedge_invweight0: edge inv. weight in qpos0            (nflexedge,)
        flex_radius: radius around primitive element              (nflex,)
        flex_edgestiffness: edge stiffness                        (nflex,)
        flex_edgedamping: edge damping                            (nflex,)
        flex_edgeequality: is edge equality constraint defined    (nflex,)
        flex_rigid: are all verices in the same body              (nflex,)
        flexedge_rigid: are both edge vertices in same body       (nflexedge,)
        flex_centered: are all vertex coordinates (0,0,0)         (nflex,)
        flex_bvhadr: address of bvh root; -1: no bvh              (nflex,)
        flex_bvhnum: number of bounding volumes                   (nflex,)
        mesh_vertadr: first vertex address                        (nmesh,)
        mesh_vertnum: number of vertices                          (nmesh,)
        mesh_faceadr: first face address                          (nmesh,)
        mesh_bvhadr: address of bvh root                          (nmesh,)
        mesh_bvhnum: number of bvh                                (nmesh,)
        mesh_graphadr: graph data address; -1: no graph           (nmesh,)
        mesh_vert: vertex positions for all meshes                (nmeshvert, 3)
        mesh_face: vertex face data                               (nmeshface, 3)
        mesh_graph: convex graph data                             (nmeshgraph,)
        mesh_pos: translation applied to asset vertices           (nmesh, 3)
        mesh_quat: rotation applied to asset vertices             (nmesh, 4)
        mesh_convex: pre-compiled convex mesh info for MJX        (nmesh,)
        mesh_texcoordadr: texcoord data address; -1: no texcoord  (nmesh,)
        mesh_texcoordnum: number of texcoord                      (nmesh,)
        mesh_texcoord: vertex texcoords for all meshes            (nmeshtexcoord, 2)
        hfield_size: (x, y, z_top, z_bottom)                      (nhfield,)
        hfield_nrow: number of rows in grid                       (nhfield,)
        hfield_ncol: number of columns in grid                    (nhfield,)
        hfield_adr: address in hfield_data                        (nhfield,)
        hfield_data: elevation data                               (nhfielddata,)
        tex_type: texture type (mjtTexture)                       (ntex,)
        tex_height: number of rows in texture image               (ntex,)
        tex_width: number of columns in texture image             (ntex,)
        tex_nchannel: number of channels in texture image         (ntex,)
        tex_adr: start address in tex_data                        (ntex,)
        tex_data: pixel values                                    (ntexdata,)
        mat_rgba: rgba                                            (nmat, 4)
        mat_texid: indices of textures; -1: none                  (nmat, mjNTEXROLE)
        pair_dim: contact dimensionality                          (npair,)
        pair_geom1: id of geom1                                   (npair,)
        pair_geom2: id of geom2                                   (npair,)
        pair_signature: body1 << 16 + body2                       (npair,)
        pair_solref: solver reference: contact normal             (npair, mjNREF)
        pair_solreffriction: solver reference: contact friction   (npair, mjNREF)
        pair_solimp: solver impedance: contact                    (npair, mjNIMP)
        pair_margin: include in solver if dist<margin-gap         (npair,)
        pair_gap: include in solver if dist<margin-gap            (npair,)
        pair_friction: tangent1, 2, spin, roll1, 2                (npair, 5)
        exclude_signature: (body1+1) << 16 + body2+1              (nexclude,)
        eq_type: constraint type (mjtEq)                          (neq,)
        eq_obj1id: id of object 1                                 (neq,)
        eq_obj2id: id of object 2                                 (neq,)
        eq_objtype: type of both objects (mjtObj)                 (neq,)
        eq_active0: initial enable/disable constraint state       (neq,)
        eq_solref: constraint solver reference                    (neq, mjNREF)
        eq_solimp: constraint solver impedance                    (neq, mjNIMP)
        eq_data: numeric data for constraint                      (neq, mjNEQDATA)
        tendon_adr: address of first object in tendon's path      (ntendon,)
        tendon_num: number of objects in tendon's path            (ntendon,)
        tendon_limited: does tendon have length limits            (ntendon,)
        tendon_solref_lim: constraint solver reference: limit     (ntendon, mjNREF)
        tendon_solimp_lim: constraint solver impedance: limit     (ntendon, mjNIMP)
        tendon_solref_fri: constraint solver reference: friction  (ntendon, mjNREF)
        tendon_solimp_fri: constraint solver impedance: friction  (ntendon, mjNIMP)
        tendon_range: tendon length limits                        (ntendon, 2)
        tendon_margin: min distance for limit detection           (ntendon,)
        tendon_stiffness: stiffness coefficient                   (ntendon,)
        tendon_damping: damping coefficient                       (ntendon,)
        tendon_frictionloss: loss due to friction                 (ntendon,)
        tendon_lengthspring: spring resting length range          (ntendon, 2)
        tendon_length0: tendon length in qpos0                    (ntendon,)
        tendon_invweight0: inv. weight in qpos0                   (ntendon,)
        tendon_hasfrictionloss: tendon has >0 frictionloss (MJX)  (ntendon,)
        wrap_type: wrap object type (mjtWrap)                     (nwrap,)
        wrap_objid: object id: geom, site, joint                  (nwrap,)
        wrap_prm: divisor, joint coef, or site id                 (nwrap,)
        wrap_inside_maxiter: maximum iterations for wrap_inside
        wrap_inside_tolerance: tolerance for wrap_inside
        wrap_inside_z_init: initialization for wrap_inside
        is_wrap_inside: spatial tendon sidesite inside geom       (nwrapinside,)
        actuator_trntype: transmission type (mjtTrn)              (nu,)
        actuator_dyntype: dynamics type (mjtDyn)                  (nu,)
        actuator_gaintype: gain type (mjtGain)                    (nu,)
        actuator_biastype: bias type (mjtBias)                    (nu,)
        actuator_trnid: transmission id: joint, tendon, site      (nu, 2)
        actuator_actadr: first activation address; -1: stateless  (nu,)
        actuator_actnum: number of activation variables           (nu,)
        actuator_group: group for visibility                      (nu,)
        actuator_ctrllimited: is control limited                  (nu,)
        actuator_forcelimited: is force limited                   (nu,)
        actuator_actlimited: is activation limited                (nu,)
        actuator_dynprm: dynamics parameters                      (nu, mjNDYN)
        actuator_gainprm: gain parameters                         (nu, mjNGAIN)
        actuator_biasprm: bias parameters                         (nu, mjNBIAS)
        actuator_actearly: step activation before force           (nu,)
        actuator_ctrlrange: range of controls                     (nu, 2)
        actuator_forcerange: range of forces                      (nu, 2)
        actuator_actrange: range of activations                   (nu, 2)
        actuator_gear: scale length and transmitted force         (nu, 6)
        actuator_cranklength: crank length for slider-crank       (nu,)
        actuator_acc0: acceleration from unit force in qpos0      (nu,)
        actuator_lengthrange: feasible actuator length range      (nu, 2)
        sensor_type: sensor type (mjtSensor)                      (nsensor,)
        sensor_datatype: numeric data type (mjtDataType)          (nsensor,)
        sensor_needstage: required compute stage (mjtStage)       (nsensor,)
        sensor_objtype: type of sensorized object (mjtObj)        (nsensor,)
        sensor_objid: id of sensorized object                     (nsensor,)
        sensor_reftype: type of reference frame (mjtObj)          (nsensor,)
        sensor_refid: id of reference frame; -1: global frame     (nsensor,)
        sensor_dim: number of scalar outputs                      (nsensor,)
        sensor_adr: address in sensor array                       (nsensor,)
        sensor_cutoff: cutoff for real and positive; 0: ignore    (nsensor,)
        numeric_adr: address of field in numeric_data             (nnumeric,)
        numeric_data: array of all numeric fields                 (nnumericdata,)
        tuple_adr: address of text in text_data                   (ntuple,)
        tuple_size: number of objects in tuple                    (ntuple,)
        tuple_objtype: array of object types in all tuples        (ntupledata,)
        tuple_objid: array of object ids in all tuples            (ntupledata,)
        tuple_objprm: array of object params in all tuples        (ntupledata,)
        key_time: key time                                        (nkey,)
        key_qpos: key position                                    (nkey, nq)
        key_qvel: key velocity                                    (nkey, nv)
        key_act: key activation                                   (nkey, na)
        key_mpos: key mocap position                              (nkey, nmocap, 3)
        key_mquat: key mocap quaternion                           (nkey, nmocap, 4)
        key_ctrl: key control                                     (nkey, nu)
        name_bodyadr: body name pointers                          (nbody,)
        name_jntadr: joint name pointers                          (njnt,)
        name_geomadr: geom name pointers                          (ngeom,)
        name_siteadr: site name pointers                          (nsite,)
        name_camadr: camera name pointers                         (ncam,)
        name_meshadr: mesh name pointers                          (nmesh,)
        name_pairadr: geom pair name pointers                     (npair,)
        name_eqadr: equality constraint name pointers             (neq,)
        name_tendonadr: tendon name pointers                      (ntendon,)
        name_actuatoradr: actuator name pointers                  (nu,)
        name_sensoradr: sensor name pointers                      (nsensor,)
        name_numericadr: numeric name pointers                    (nnumeric,)
        name_tupleadr: tuple name pointers                        (ntuple,)
        name_keyadr: keyframe name pointers                       (nkey,)
        names: names of all objects, 0-terminated                 (nnames,)
      
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'nq': Field(name='nq',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nv': Field(name='nv',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nu': Field(name='nu',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'na': Field(name='na',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nbody': Field(name='nbody',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nbvh': Field(name='nbvh',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nbvhstatic': Field(name='nbvhstatic',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nbvhdynamic': Field(name='nbvhdynamic',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'njnt': Field(name='njnt',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ngeom': Field(name='ngeom',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nsite': Field(name='nsite',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ncam': Field(name='ncam',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nlight': Field(name='nlight',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflex': Field(name='nflex',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflexvert': Field(name='nflexvert',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflexedge': Field(name='nflexedge',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflexelem': Field(name='nflexelem',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflexelemdata': Field(name='nflexelemdata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflexshelldata': Field(name='nflexshelldata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflexevpair': Field(name='nflexevpair',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nflextexcoord': Field(name='nflextexcoord',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'nmesh': Field(name='nmesh',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshvert': Field(name='nmeshvert',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshnormal': Field(name='nmeshnormal',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshtexcoord': Field(name='nmeshtexcoord',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshface': Field(name='nmeshface',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshgraph': Field(name='nmeshgraph',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nhfield': Field(name='nhfield',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nhfielddata': Field(name='nhfielddata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntex': Field(name='ntex',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntexdata': Field(name='ntexdata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmat': Field(name='nmat',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'npair': Field(name='npair',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nexclude': Field(name='nexclude',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'neq': Field(name='neq',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntendon': Field(name='ntendon',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nwrap': Field(name='nwrap',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nsensor': Field(name='nsensor',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nnumeric': Field(name='nnumeric',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntuple': Field(name='ntuple',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nkey': Field(name='nkey',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmocap': Field(name='nmocap',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nM': Field(name='nM',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nB': Field(name='nB',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nC': Field(name='nC',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nD': Field(name='nD',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nJmom': Field(name='nJmom',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntree': Field(name='ntree',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ngravcomp': Field(name='ngravcomp',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nuserdata': Field(name='nuserdata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nsensordata': Field(name='nsensordata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'narena': Field(name='narena',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'opt': Field(name='opt',type=<class 'mujoco.mjx._src.types.Option'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'stat': Field(name='stat',type=<class 'mujoco.mjx._src.types.Statistic'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qpos0': Field(name='qpos0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qpos_spring': Field(name='qpos_spring',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_parentid': Field(name='body_parentid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_mocapid': Field(name='body_mocapid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_rootid': Field(name='body_rootid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_weldid': Field(name='body_weldid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_jntnum': Field(name='body_jntnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_jntadr': Field(name='body_jntadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_sameframe': Field(name='body_sameframe',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_dofnum': Field(name='body_dofnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_dofadr': Field(name='body_dofadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_treeid': Field(name='body_treeid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_geomnum': Field(name='body_geomnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_geomadr': Field(name='body_geomadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_simple': Field(name='body_simple',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_pos': Field(name='body_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_quat': Field(name='body_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_ipos': Field(name='body_ipos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_iquat': Field(name='body_iquat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_mass': Field(name='body_mass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_subtreemass': Field(name='body_subtreemass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_inertia': Field(name='body_inertia',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_gravcomp': Field(name='body_gravcomp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_margin': Field(name='body_margin',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_contype': Field(name='body_contype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_conaffinity': Field(name='body_conaffinity',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_bvhadr': Field(name='body_bvhadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'body_bvhnum': Field(name='body_bvhnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'bvh_child': Field(name='bvh_child',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'bvh_nodeid': Field(name='bvh_nodeid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'bvh_aabb': Field(name='bvh_aabb',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'body_invweight0': Field(name='body_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_type': Field(name='jnt_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_qposadr': Field(name='jnt_qposadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_dofadr': Field(name='jnt_dofadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_bodyid': Field(name='jnt_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_limited': Field(name='jnt_limited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_actfrclimited': Field(name='jnt_actfrclimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_actgravcomp': Field(name='jnt_actgravcomp',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_solref': Field(name='jnt_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_solimp': Field(name='jnt_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_pos': Field(name='jnt_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_axis': Field(name='jnt_axis',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_stiffness': Field(name='jnt_stiffness',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_range': Field(name='jnt_range',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_actfrcrange': Field(name='jnt_actfrcrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_margin': Field(name='jnt_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_bodyid': Field(name='dof_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_jntid': Field(name='dof_jntid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_parentid': Field(name='dof_parentid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_treeid': Field(name='dof_treeid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_Madr': Field(name='dof_Madr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_simplenum': Field(name='dof_simplenum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_solref': Field(name='dof_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_solimp': Field(name='dof_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_frictionloss': Field(name='dof_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_hasfrictionloss': Field(name='dof_hasfrictionloss',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'dof_armature': Field(name='dof_armature',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_damping': Field(name='dof_damping',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_invweight0': Field(name='dof_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_M0': Field(name='dof_M0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_type': Field(name='geom_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_contype': Field(name='geom_contype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_conaffinity': Field(name='geom_conaffinity',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_condim': Field(name='geom_condim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_bodyid': Field(name='geom_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_sameframe': Field(name='geom_sameframe',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_dataid': Field(name='geom_dataid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_group': Field(name='geom_group',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_matid': Field(name='geom_matid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_priority': Field(name='geom_priority',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_solmix': Field(name='geom_solmix',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_solref': Field(name='geom_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_solimp': Field(name='geom_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_size': Field(name='geom_size',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_aabb': Field(name='geom_aabb',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_rbound': Field(name='geom_rbound',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_rbound_hfield': Field(name='geom_rbound_hfield',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'geom_pos': Field(name='geom_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_quat': Field(name='geom_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_friction': Field(name='geom_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_margin': Field(name='geom_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_gap': Field(name='geom_gap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_fluid': Field(name='geom_fluid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_rgba': Field(name='geom_rgba',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_type': Field(name='site_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_bodyid': Field(name='site_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_sameframe': Field(name='site_sameframe',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_size': Field(name='site_size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_pos': Field(name='site_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_quat': Field(name='site_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_mode': Field(name='cam_mode',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_bodyid': Field(name='cam_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_targetbodyid': Field(name='cam_targetbodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_pos': Field(name='cam_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_quat': Field(name='cam_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_poscom0': Field(name='cam_poscom0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_pos0': Field(name='cam_pos0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_mat0': Field(name='cam_mat0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_fovy': Field(name='cam_fovy',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_resolution': Field(name='cam_resolution',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_sensorsize': Field(name='cam_sensorsize',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_intrinsic': Field(name='cam_intrinsic',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_mode': Field(name='light_mode',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_bodyid': Field(name='light_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'light_targetbodyid': Field(name='light_targetbodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'light_directional': Field(name='light_directional',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_castshadow': Field(name='light_castshadow',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_pos': Field(name='light_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_dir': Field(name='light_dir',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_poscom0': Field(name='light_poscom0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_pos0': Field(name='light_pos0',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_dir0': Field(name='light_dir0',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_cutoff': Field(name='light_cutoff',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_contype': Field(name='flex_contype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_conaffinity': Field(name='flex_conaffinity',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_condim': Field(name='flex_condim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_priority': Field(name='flex_priority',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_solmix': Field(name='flex_solmix',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_solref': Field(name='flex_solref',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_solimp': Field(name='flex_solimp',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_friction': Field(name='flex_friction',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_margin': Field(name='flex_margin',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_gap': Field(name='flex_gap',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_internal': Field(name='flex_internal',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_selfcollide': Field(name='flex_selfcollide',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_activelayers': Field(name='flex_activelayers',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_dim': Field(name='flex_dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_vertadr': Field(name='flex_vertadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_vertnum': Field(name='flex_vertnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_edgeadr': Field(name='flex_edgeadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_edgenum': Field(name='flex_edgenum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_elemadr': Field(name='flex_elemadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_elemnum': Field(name='flex_elemnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_elemdataadr': Field(name='flex_elemdataadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_evpairadr': Field(name='flex_evpairadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_evpairnum': Field(name='flex_evpairnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_vertbodyid': Field(name='flex_vertbodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_edge': Field(name='flex_edge',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_elem': Field(name='flex_elem',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_elemlayer': Field(name='flex_elemlayer',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_evpair': Field(name='flex_evpair',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_vert': Field(name='flex_vert',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_length0': Field(name='flexedge_length0',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_invweight0': Field(name='flexedge_invweight0',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_radius': Field(name='flex_radius',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_edgestiffness': Field(name='flex_edgestiffness',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_edgedamping': Field(name='flex_edgedamping',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_edgeequality': Field(name='flex_edgeequality',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_rigid': Field(name='flex_rigid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flexedge_rigid': Field(name='flexedge_rigid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_centered': Field(name='flex_centered',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_bvhadr': Field(name='flex_bvhadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'flex_bvhnum': Field(name='flex_bvhnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'mesh_vertadr': Field(name='mesh_vertadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_vertnum': Field(name='mesh_vertnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_faceadr': Field(name='mesh_faceadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_bvhadr': Field(name='mesh_bvhadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_bvhnum': Field(name='mesh_bvhnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_graphadr': Field(name='mesh_graphadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_vert': Field(name='mesh_vert',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_face': Field(name='mesh_face',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_graph': Field(name='mesh_graph',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_pos': Field(name='mesh_pos',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_quat': Field(name='mesh_quat',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_convex': Field(name='mesh_convex',type=typing.Tuple[mujoco.mjx._src.types.ConvexMesh, ...],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'mesh_texcoordadr': Field(name='mesh_texcoordadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_texcoordnum': Field(name='mesh_texcoordnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_texcoord': Field(name='mesh_texcoord',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_size': Field(name='hfield_size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_nrow': Field(name='hfield_nrow',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_ncol': Field(name='hfield_ncol',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_adr': Field(name='hfield_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_data': Field(name='hfield_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_type': Field(name='tex_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_height': Field(name='tex_height',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_width': Field(name='tex_width',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_nchannel': Field(name='tex_nchannel',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_adr': Field(name='tex_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_data': Field(name='tex_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat_rgba': Field(name='mat_rgba',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat_texid': Field(name='mat_texid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_dim': Field(name='pair_dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_geom1': Field(name='pair_geom1',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_geom2': Field(name='pair_geom2',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_signature': Field(name='pair_signature',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_solref': Field(name='pair_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_solreffriction': Field(name='pair_solreffriction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_solimp': Field(name='pair_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_margin': Field(name='pair_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_gap': Field(name='pair_gap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_friction': Field(name='pair_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'exclude_signature': Field(name='exclude_signature',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_type': Field(name='eq_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_obj1id': Field(name='eq_obj1id',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_obj2id': Field(name='eq_obj2id',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_objtype': Field(name='eq_objtype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_active0': Field(name='eq_active0',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_solref': Field(name='eq_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_solimp': Field(name='eq_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_data': Field(name='eq_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_adr': Field(name='tendon_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_num': Field(name='tendon_num',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_limited': Field(name='tendon_limited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solref_lim': Field(name='tendon_solref_lim',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solimp_lim': Field(name='tendon_solimp_lim',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solref_fri': Field(name='tendon_solref_fri',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solimp_fri': Field(name='tendon_solimp_fri',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_range': Field(name='tendon_range',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_margin': Field(name='tendon_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_stiffness': Field(name='tendon_stiffness',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_damping': Field(name='tendon_damping',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_frictionloss': Field(name='tendon_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_lengthspring': Field(name='tendon_lengthspring',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_length0': Field(name='tendon_length0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_invweight0': Field(name='tendon_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_hasfrictionloss': Field(name='tendon_hasfrictionloss',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'wrap_type': Field(name='wrap_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_objid': Field(name='wrap_objid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_prm': Field(name='wrap_prm',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_inside_maxiter': Field(name='wrap_inside_maxiter',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'wrap_inside_tolerance': Field(name='wrap_inside_tolerance',type=<class 'float'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'wrap_inside_z_init': Field(name='wrap_inside_z_init',type=<class 'float'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'is_wrap_inside': Field(name='is_wrap_inside',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'actuator_trntype': Field(name='actuator_trntype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_dyntype': Field(name='actuator_dyntype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_gaintype': Field(name='actuator_gaintype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_biastype': Field(name='actuator_biastype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_trnid': Field(name='actuator_trnid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actadr': Field(name='actuator_actadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actnum': Field(name='actuator_actnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_group': Field(name='actuator_group',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_ctrllimited': Field(name='actuator_ctrllimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_forcelimited': Field(name='actuator_forcelimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actlimited': Field(name='actuator_actlimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_dynprm': Field(name='actuator_dynprm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_gainprm': Field(name='actuator_gainprm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_biasprm': Field(name='actuator_biasprm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actearly': Field(name='actuator_actearly',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_ctrlrange': Field(name='actuator_ctrlrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_forcerange': Field(name='actuator_forcerange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actrange': Field(name='actuator_actrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_gear': Field(name='actuator_gear',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_cranklength': Field(name='actuator_cranklength',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_acc0': Field(name='actuator_acc0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_lengthrange': Field(name='actuator_lengthrange',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_plugin': Field(name='actuator_plugin',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'sensor_type': Field(name='sensor_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_datatype': Field(name='sensor_datatype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_needstage': Field(name='sensor_needstage',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_objtype': Field(name='sensor_objtype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_objid': Field(name='sensor_objid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_reftype': Field(name='sensor_reftype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_refid': Field(name='sensor_refid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_dim': Field(name='sensor_dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_adr': Field(name='sensor_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_cutoff': Field(name='sensor_cutoff',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'numeric_adr': Field(name='numeric_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'numeric_data': Field(name='numeric_data',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_adr': Field(name='tuple_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_size': Field(name='tuple_size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_objtype': Field(name='tuple_objtype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_objid': Field(name='tuple_objid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_objprm': Field(name='tuple_objprm',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_time': Field(name='key_time',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_qpos': Field(name='key_qpos',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_qvel': Field(name='key_qvel',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_act': Field(name='key_act',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_mpos': Field(name='key_mpos',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_mquat': Field(name='key_mquat',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_ctrl': Field(name='key_ctrl',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_bodyadr': Field(name='name_bodyadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_jntadr': Field(name='name_jntadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_geomadr': Field(name='name_geomadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_siteadr': Field(name='name_siteadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_camadr': Field(name='name_camadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_meshadr': Field(name='name_meshadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_hfieldadr': Field(name='name_hfieldadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_pairadr': Field(name='name_pairadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_eqadr': Field(name='name_eqadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_tendonadr': Field(name='name_tendonadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_actuatoradr': Field(name='name_actuatoradr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_sensoradr': Field(name='name_sensoradr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_numericadr': Field(name='name_numericadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_tupleadr': Field(name='name_tupleadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_keyadr': Field(name='name_keyadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'names': Field(name='names',type=<class 'bytes'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), '_sizes': Field(name='_sizes',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = ('nq', 'nv', 'nu', 'na', 'nbody', 'nbvh', 'nbvhstatic', 'nbvhdynamic', 'njnt', 'ngeom', 'nsite', 'ncam', 'nlight', 'nflex', 'nflexvert', 'nflexedge', 'nflexelem', 'nflexelemdata', 'nflexshelldata', 'nflexevpair', 'nflextexcoord', 'nmesh', 'nmeshvert', 'nmeshnormal', 'nmeshtexcoord', 'nmeshface', 'nmeshgraph', 'nhfield', 'nhfielddata', 'ntex', 'ntexdata', 'nmat', 'npair', 'nexclude', 'neq', 'ntendon', 'nwrap', 'nsensor', 'nnumeric', 'ntuple', 'nkey', 'nmocap', 'nM', 'nB', 'nC', 'nD', 'nJmom', 'ntree', 'ngravcomp', 'nuserdata', 'nsensordata', 'narena', 'opt', 'stat', 'qpos0', 'qpos_spring', 'body_parentid', 'body_mocapid', 'body_rootid', 'body_weldid', 'body_jntnum', 'body_jntadr', 'body_sameframe', 'body_dofnum', 'body_dofadr', 'body_treeid', 'body_geomnum', 'body_geomadr', 'body_simple', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass', 'body_subtreemass', 'body_inertia', 'body_gravcomp', 'body_margin', 'body_contype', 'body_conaffinity', 'body_bvhadr', 'body_bvhnum', 'bvh_child', 'bvh_nodeid', 'bvh_aabb', 'body_invweight0', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_bodyid', 'jnt_limited', 'jnt_actfrclimited', 'jnt_actgravcomp', 'jnt_solref', 'jnt_solimp', 'jnt_pos', 'jnt_axis', 'jnt_stiffness', 'jnt_range', 'jnt_actfrcrange', 'jnt_margin', 'dof_bodyid', 'dof_jntid', 'dof_parentid', 'dof_treeid', 'dof_Madr', 'dof_simplenum', 'dof_solref', 'dof_solimp', 'dof_frictionloss', 'dof_hasfrictionloss', 'dof_armature', 'dof_damping', 'dof_invweight0', 'dof_M0', 'geom_type', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_bodyid', 'geom_sameframe', 'geom_dataid', 'geom_group', 'geom_matid', 'geom_priority', 'geom_solmix', 'geom_solref', 'geom_solimp', 'geom_size', 'geom_aabb', 'geom_rbound', 'geom_rbound_hfield', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_margin', 'geom_gap', 'geom_fluid', 'geom_rgba', 'site_type', 'site_bodyid', 'site_sameframe', 'site_size', 'site_pos', 'site_quat', 'cam_mode', 'cam_bodyid', 'cam_targetbodyid', 'cam_pos', 'cam_quat', 'cam_poscom0', 'cam_pos0', 'cam_mat0', 'cam_fovy', 'cam_resolution', 'cam_sensorsize', 'cam_intrinsic', 'light_mode', 'light_bodyid', 'light_targetbodyid', 'light_directional', 'light_castshadow', 'light_pos', 'light_dir', 'light_poscom0', 'light_pos0', 'light_dir0', 'light_cutoff', 'flex_contype', 'flex_conaffinity', 'flex_condim', 'flex_priority', 'flex_solmix', 'flex_solref', 'flex_solimp', 'flex_friction', 'flex_margin', 'flex_gap', 'flex_internal', 'flex_selfcollide', 'flex_activelayers', 'flex_dim', 'flex_vertadr', 'flex_vertnum', 'flex_edgeadr', 'flex_edgenum', 'flex_elemadr', 'flex_elemnum', 'flex_elemdataadr', 'flex_evpairadr', 'flex_evpairnum', 'flex_vertbodyid', 'flex_edge', 'flex_elem', 'flex_elemlayer', 'flex_evpair', 'flex_vert', 'flexedge_length0', 'flexedge_invweight0', 'flex_radius', 'flex_edgestiffness', 'flex_edgedamping', 'flex_edgeequality', 'flex_rigid', 'flexedge_rigid', 'flex_centered', 'flex_bvhadr', 'flex_bvhnum', 'mesh_vertadr', 'mesh_vertnum', 'mesh_faceadr', 'mesh_bvhadr', 'mesh_bvhnum', 'mesh_graphadr', 'mesh_vert', 'mesh_face', 'mesh_graph', 'mesh_pos', 'mesh_quat', 'mesh_convex', 'mesh_texcoordadr', 'mesh_texcoordnum', 'mesh_texcoord', 'hfield_size', 'hfield_nrow', 'hfield_ncol', 'hfield_adr', 'hfield_data', 'tex_type', 'tex_height', 'tex_width', 'tex_nchannel', 'tex_adr', 'tex_data', 'mat_rgba', 'mat_texid', 'pair_dim', 'pair_geom1', 'pair_geom2', 'pair_signature', 'pair_solref', 'pair_solreffriction', 'pair_solimp', 'pair_margin', 'pair_gap', 'pair_friction', 'exclude_signature', 'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_objtype', 'eq_active0', 'eq_solref', 'eq_solimp', 'eq_data', 'tendon_adr', 'tendon_num', 'tendon_limited', 'tendon_solref_lim', 'tendon_solimp_lim', 'tendon_solref_fri', 'tendon_solimp_fri', 'tendon_range', 'tendon_margin', 'tendon_stiffness', 'tendon_damping', 'tendon_frictionloss', 'tendon_lengthspring', 'tendon_length0', 'tendon_invweight0', 'tendon_hasfrictionloss', 'wrap_type', 'wrap_objid', 'wrap_prm', 'wrap_inside_maxiter', 'wrap_inside_tolerance', 'wrap_inside_z_init', 'is_wrap_inside', 'actuator_trntype', 'actuator_dyntype', 'actuator_gaintype', 'actuator_biastype', 'actuator_trnid', 'actuator_actadr', 'actuator_actnum', 'actuator_group', 'actuator_ctrllimited', 'actuator_forcelimited', 'actuator_actlimited', 'actuator_dynprm', 'actuator_gainprm', 'actuator_biasprm', 'actuator_actearly', 'actuator_ctrlrange', 'actuator_forcerange', 'actuator_actrange', 'actuator_gear', 'actuator_cranklength', 'actuator_acc0', 'actuator_lengthrange', 'actuator_plugin', 'sensor_type', 'sensor_datatype', 'sensor_needstage', 'sensor_objtype', 'sensor_objid', 'sensor_reftype', 'sensor_refid', 'sensor_dim', 'sensor_adr', 'sensor_cutoff', 'numeric_adr', 'numeric_data', 'tuple_adr', 'tuple_size', 'tuple_objtype', 'tuple_objid', 'tuple_objprm', 'key_time', 'key_qpos', 'key_qvel', 'key_act', 'key_mpos', 'key_mquat', 'key_ctrl', 'name_bodyadr', 'name_jntadr', 'name_geomadr', 'name_siteadr', 'name_camadr', 'name_meshadr', 'name_hfieldadr', 'name_pairadr', 'name_eqadr', 'name_tendonadr', 'name_actuatoradr', 'name_sensoradr', 'name_numericadr', 'name_tupleadr', 'name_keyadr', 'names', '_sizes')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, nq: int, nv: int, nu: int, na: int, nbody: int, nbvh: int, nbvhstatic: int, nbvhdynamic: int, njnt: int, ngeom: int, nsite: int, ncam: int, nlight: int, nflex: int, nflexvert: int, nflexedge: int, nflexelem: int, nflexelemdata: int, nflexshelldata: int, nflexevpair: int, nflextexcoord: int, nmesh: int, nmeshvert: int, nmeshnormal: int, nmeshtexcoord: int, nmeshface: int, nmeshgraph: int, nhfield: int, nhfielddata: int, ntex: int, ntexdata: int, nmat: int, npair: int, nexclude: int, neq: int, ntendon: int, nwrap: int, nsensor: int, nnumeric: int, ntuple: int, nkey: int, nmocap: int, nM: int, nB: int, nC: int, nD: int, nJmom: int, ntree: int, ngravcomp: int, nuserdata: int, nsensordata: int, narena: int, opt: Option, stat: Statistic, qpos0: jax.Array, qpos_spring: jax.Array, body_parentid: numpy.ndarray, body_mocapid: numpy.ndarray, body_rootid: numpy.ndarray, body_weldid: numpy.ndarray, body_jntnum: numpy.ndarray, body_jntadr: numpy.ndarray, body_sameframe: numpy.ndarray, body_dofnum: numpy.ndarray, body_dofadr: numpy.ndarray, body_treeid: numpy.ndarray, body_geomnum: numpy.ndarray, body_geomadr: numpy.ndarray, body_simple: numpy.ndarray, body_pos: jax.Array, body_quat: jax.Array, body_ipos: jax.Array, body_iquat: jax.Array, body_mass: jax.Array, body_subtreemass: jax.Array, body_inertia: jax.Array, body_gravcomp: jax.Array, body_margin: numpy.ndarray, body_contype: numpy.ndarray, body_conaffinity: numpy.ndarray, body_bvhadr: numpy.ndarray, body_bvhnum: numpy.ndarray, bvh_child: numpy.ndarray, bvh_nodeid: numpy.ndarray, bvh_aabb: numpy.ndarray, body_invweight0: jax.Array, jnt_type: numpy.ndarray, jnt_qposadr: numpy.ndarray, jnt_dofadr: numpy.ndarray, jnt_bodyid: numpy.ndarray, jnt_limited: numpy.ndarray, jnt_actfrclimited: numpy.ndarray, jnt_actgravcomp: numpy.ndarray, jnt_solref: jax.Array, jnt_solimp: jax.Array, jnt_pos: jax.Array, jnt_axis: jax.Array, jnt_stiffness: jax.Array, jnt_range: jax.Array, jnt_actfrcrange: jax.Array, jnt_margin: jax.Array, dof_bodyid: numpy.ndarray, dof_jntid: numpy.ndarray, dof_parentid: numpy.ndarray, dof_treeid: numpy.ndarray, dof_Madr: numpy.ndarray, dof_simplenum: numpy.ndarray, dof_solref: jax.Array, dof_solimp: jax.Array, dof_frictionloss: jax.Array, dof_hasfrictionloss: numpy.ndarray, dof_armature: jax.Array, dof_damping: jax.Array, dof_invweight0: jax.Array, dof_M0: jax.Array, geom_type: numpy.ndarray, geom_contype: numpy.ndarray, geom_conaffinity: numpy.ndarray, geom_condim: numpy.ndarray, geom_bodyid: numpy.ndarray, geom_sameframe: numpy.ndarray, geom_dataid: numpy.ndarray, geom_group: numpy.ndarray, geom_matid: jax.Array, geom_priority: numpy.ndarray, geom_solmix: jax.Array, geom_solref: jax.Array, geom_solimp: jax.Array, geom_size: jax.Array, geom_aabb: numpy.ndarray, geom_rbound: jax.Array, geom_rbound_hfield: numpy.ndarray, geom_pos: jax.Array, geom_quat: jax.Array, geom_friction: jax.Array, geom_margin: jax.Array, geom_gap: jax.Array, geom_fluid: numpy.ndarray, geom_rgba: jax.Array, site_type: numpy.ndarray, site_bodyid: numpy.ndarray, site_sameframe: numpy.ndarray, site_size: numpy.ndarray, site_pos: jax.Array, site_quat: jax.Array, cam_mode: numpy.ndarray, cam_bodyid: numpy.ndarray, cam_targetbodyid: numpy.ndarray, cam_pos: jax.Array, cam_quat: jax.Array, cam_poscom0: jax.Array, cam_pos0: jax.Array, cam_mat0: jax.Array, cam_fovy: numpy.ndarray, cam_resolution: numpy.ndarray, cam_sensorsize: numpy.ndarray, cam_intrinsic: numpy.ndarray, light_mode: numpy.ndarray, light_bodyid: numpy.ndarray, light_targetbodyid: numpy.ndarray, light_directional: jax.Array, light_castshadow: jax.Array, light_pos: jax.Array, light_dir: jax.Array, light_poscom0: jax.Array, light_pos0: numpy.ndarray, light_dir0: numpy.ndarray, light_cutoff: jax.Array, flex_contype: numpy.ndarray, flex_conaffinity: numpy.ndarray, flex_condim: numpy.ndarray, flex_priority: numpy.ndarray, flex_solmix: numpy.ndarray, flex_solref: numpy.ndarray, flex_solimp: numpy.ndarray, flex_friction: numpy.ndarray, flex_margin: numpy.ndarray, flex_gap: numpy.ndarray, flex_internal: numpy.ndarray, flex_selfcollide: numpy.ndarray, flex_activelayers: numpy.ndarray, flex_dim: numpy.ndarray, flex_vertadr: numpy.ndarray, flex_vertnum: numpy.ndarray, flex_edgeadr: numpy.ndarray, flex_edgenum: numpy.ndarray, flex_elemadr: numpy.ndarray, flex_elemnum: numpy.ndarray, flex_elemdataadr: numpy.ndarray, flex_evpairadr: numpy.ndarray, flex_evpairnum: numpy.ndarray, flex_vertbodyid: numpy.ndarray, flex_edge: numpy.ndarray, flex_elem: numpy.ndarray, flex_elemlayer: numpy.ndarray, flex_evpair: numpy.ndarray, flex_vert: numpy.ndarray, flexedge_length0: numpy.ndarray, flexedge_invweight0: numpy.ndarray, flex_radius: numpy.ndarray, flex_edgestiffness: numpy.ndarray, flex_edgedamping: numpy.ndarray, flex_edgeequality: numpy.ndarray, flex_rigid: numpy.ndarray, flexedge_rigid: numpy.ndarray, flex_centered: numpy.ndarray, flex_bvhadr: numpy.ndarray, flex_bvhnum: numpy.ndarray, mesh_vertadr: numpy.ndarray, mesh_vertnum: numpy.ndarray, mesh_faceadr: numpy.ndarray, mesh_bvhadr: numpy.ndarray, mesh_bvhnum: numpy.ndarray, mesh_graphadr: numpy.ndarray, mesh_vert: numpy.ndarray, mesh_face: numpy.ndarray, mesh_graph: numpy.ndarray, mesh_pos: numpy.ndarray, mesh_quat: numpy.ndarray, mesh_convex: typing.Tuple[mujoco.mjx._src.types.ConvexMesh, ...], mesh_texcoordadr: numpy.ndarray, mesh_texcoordnum: numpy.ndarray, mesh_texcoord: numpy.ndarray, hfield_size: numpy.ndarray, hfield_nrow: numpy.ndarray, hfield_ncol: numpy.ndarray, hfield_adr: numpy.ndarray, hfield_data: jax.Array, tex_type: numpy.ndarray, tex_height: numpy.ndarray, tex_width: numpy.ndarray, tex_nchannel: numpy.ndarray, tex_adr: numpy.ndarray, tex_data: jax.Array, mat_rgba: jax.Array, mat_texid: numpy.ndarray, pair_dim: numpy.ndarray, pair_geom1: numpy.ndarray, pair_geom2: numpy.ndarray, pair_signature: numpy.ndarray, pair_solref: jax.Array, pair_solreffriction: jax.Array, pair_solimp: jax.Array, pair_margin: jax.Array, pair_gap: jax.Array, pair_friction: jax.Array, exclude_signature: numpy.ndarray, eq_type: numpy.ndarray, eq_obj1id: numpy.ndarray, eq_obj2id: numpy.ndarray, eq_objtype: numpy.ndarray, eq_active0: numpy.ndarray, eq_solref: jax.Array, eq_solimp: jax.Array, eq_data: jax.Array, tendon_adr: numpy.ndarray, tendon_num: numpy.ndarray, tendon_limited: numpy.ndarray, tendon_solref_lim: jax.Array, tendon_solimp_lim: jax.Array, tendon_solref_fri: jax.Array, tendon_solimp_fri: jax.Array, tendon_range: jax.Array, tendon_margin: jax.Array, tendon_stiffness: jax.Array, tendon_damping: jax.Array, tendon_frictionloss: jax.Array, tendon_lengthspring: jax.Array, tendon_length0: jax.Array, tendon_invweight0: jax.Array, tendon_hasfrictionloss: numpy.ndarray, wrap_type: numpy.ndarray, wrap_objid: numpy.ndarray, wrap_prm: numpy.ndarray, wrap_inside_maxiter: int, wrap_inside_tolerance: float, wrap_inside_z_init: float, is_wrap_inside: numpy.ndarray, actuator_trntype: numpy.ndarray, actuator_dyntype: numpy.ndarray, actuator_gaintype: numpy.ndarray, actuator_biastype: numpy.ndarray, actuator_trnid: numpy.ndarray, actuator_actadr: numpy.ndarray, actuator_actnum: numpy.ndarray, actuator_group: numpy.ndarray, actuator_ctrllimited: numpy.ndarray, actuator_forcelimited: numpy.ndarray, actuator_actlimited: numpy.ndarray, actuator_dynprm: jax.Array, actuator_gainprm: jax.Array, actuator_biasprm: jax.Array, actuator_actearly: numpy.ndarray, actuator_ctrlrange: jax.Array, actuator_forcerange: jax.Array, actuator_actrange: jax.Array, actuator_gear: jax.Array, actuator_cranklength: numpy.ndarray, actuator_acc0: jax.Array, actuator_lengthrange: numpy.ndarray, actuator_plugin: numpy.ndarray, sensor_type: numpy.ndarray, sensor_datatype: numpy.ndarray, sensor_needstage: numpy.ndarray, sensor_objtype: numpy.ndarray, sensor_objid: numpy.ndarray, sensor_reftype: numpy.ndarray, sensor_refid: numpy.ndarray, sensor_dim: numpy.ndarray, sensor_adr: numpy.ndarray, sensor_cutoff: numpy.ndarray, numeric_adr: numpy.ndarray, numeric_data: numpy.ndarray, tuple_adr: numpy.ndarray, tuple_size: numpy.ndarray, tuple_objtype: numpy.ndarray, tuple_objid: numpy.ndarray, tuple_objprm: numpy.ndarray, key_time: numpy.ndarray, key_qpos: numpy.ndarray, key_qvel: numpy.ndarray, key_act: numpy.ndarray, key_mpos: numpy.ndarray, key_mquat: numpy.ndarray, key_ctrl: numpy.ndarray, name_bodyadr: numpy.ndarray, name_jntadr: numpy.ndarray, name_geomadr: numpy.ndarray, name_siteadr: numpy.ndarray, name_camadr: numpy.ndarray, name_meshadr: numpy.ndarray, name_hfieldadr: numpy.ndarray, name_pairadr: numpy.ndarray, name_eqadr: numpy.ndarray, name_tendonadr: numpy.ndarray, name_actuatoradr: numpy.ndarray, name_sensoradr: numpy.ndarray, name_numericadr: numpy.ndarray, name_tupleadr: numpy.ndarray, name_keyadr: numpy.ndarray, names: bytes, _sizes: jax.Array) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
    def bind(self, obj: typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin, collections.abc.Iterable[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]]]) -> mujoco.mjx._src.support.BindModel:
        """
        Bind a Mujoco spec to an MJX Model.
        """
class ObjType(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Type of object.
    
      Members:
        UNKNOWN: unknown object type
        BODY: body
        XBODY: body, used to access regular frame instead of i-frame
        GEOM: geom
        SITE: site
        CAMERA: camera
      
    """
    BODY: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_BODY: 1>
    CAMERA: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_CAMERA: 7>
    GEOM: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_GEOM: 5>
    SITE: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_SITE: 6>
    UNKNOWN: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_UNKNOWN: 0>
    XBODY: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_XBODY: 2>
    __dataclass_fields__: typing.ClassVar[dict] = {}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = tuple()
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class Option(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Physics options.
    
      Attributes:
        timestep:          timestep
        apirate:           update rate for remote API (Hz) (not used)
        impratio:          ratio of friction-to-normal contact impedance
        tolerance:         main solver tolerance
        ls_tolerance:      CG/Newton linesearch tolerance
        noslip_tolerance:  noslip solver tolerance (not used)
        ccd_tolerance:     CCD solver tolerance (not used)
        gravity:           gravitational acceleration                 (3,)
        wind:              wind (for lift, drag and viscosity)
        magnetic:          global magnetic flux (not used)
        density:           density of medium
        viscosity:         viscosity of medium
        o_margin:          contact solver override: margin (not used)
        o_solref:          contact solver override: solref (not used)
        o_solimp:          contact solver override: solimp (not used)
        o_friction[5]:     contact solver override: friction (not used)
        has_fluid_params:  automatically set by mjx if wind/density/viscosity are
          nonzero. Not used by mj
        integrator:        integration mode
        cone:              type of friction cone
        jacobian:          matrix layout for mass matrices (dense or sparse)
                           (note that this is different from MuJoCo, where jacobian
                           specifies whether efc_J and its accompanying matrices
                           are dense or sparse.
        solver:            solver algorithm
        iterations:        number of main solver iterations
        ls_iterations:     maximum number of CG/Newton linesearch iterations
        noslip_iterations: maximum number of noslip solver iterations (not used)
        ccd_iterations:    maximum number of CCD solver iterations (not used)
        disableflags:      bit flags for disabling standard features
        enableflags:       bit flags for enabling optional features (not used)
        disableactuator:   bit flags for disabling actuators by group id (not used)
        sdf_initpoints:    number of starting points for gradient descent (not used)
        sdf_iterations:    max number of iterations for gradient descent (not used)
      
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'timestep': Field(name='timestep',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'apirate': Field(name='apirate',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'impratio': Field(name='impratio',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tolerance': Field(name='tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ls_tolerance': Field(name='ls_tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'noslip_tolerance': Field(name='noslip_tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ccd_tolerance': Field(name='ccd_tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'gravity': Field(name='gravity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wind': Field(name='wind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'magnetic': Field(name='magnetic',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'density': Field(name='density',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'viscosity': Field(name='viscosity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_margin': Field(name='o_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_solref': Field(name='o_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_solimp': Field(name='o_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_friction': Field(name='o_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'has_fluid_params': Field(name='has_fluid_params',type=<class 'bool'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mjx'}),kw_only=False,_field_type=_FIELD), 'integrator': Field(name='integrator',type=<enum 'IntegratorType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cone': Field(name='cone',type=<enum 'ConeType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jacobian': Field(name='jacobian',type=<enum 'JacobianType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solver': Field(name='solver',type=<enum 'SolverType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'iterations': Field(name='iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ls_iterations': Field(name='ls_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'noslip_iterations': Field(name='noslip_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'ccd_iterations': Field(name='ccd_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'disableflags': Field(name='disableflags',type=<flag 'DisableBit'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'enableflags': Field(name='enableflags',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'disableactuator': Field(name='disableactuator',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sdf_initpoints': Field(name='sdf_initpoints',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD), 'sdf_iterations': Field(name='sdf_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'restricted_to': 'mujoco'}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = ('timestep', 'apirate', 'impratio', 'tolerance', 'ls_tolerance', 'noslip_tolerance', 'ccd_tolerance', 'gravity', 'wind', 'magnetic', 'density', 'viscosity', 'o_margin', 'o_solref', 'o_solimp', 'o_friction', 'has_fluid_params', 'integrator', 'cone', 'jacobian', 'solver', 'iterations', 'ls_iterations', 'noslip_iterations', 'ccd_iterations', 'disableflags', 'enableflags', 'disableactuator', 'sdf_initpoints', 'sdf_iterations')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, timestep: jax.Array, apirate: jax.Array, impratio: jax.Array, tolerance: jax.Array, ls_tolerance: jax.Array, noslip_tolerance: jax.Array, ccd_tolerance: jax.Array, gravity: jax.Array, wind: jax.Array, magnetic: jax.Array, density: jax.Array, viscosity: jax.Array, o_margin: jax.Array, o_solref: jax.Array, o_solimp: jax.Array, o_friction: jax.Array, has_fluid_params: bool, integrator: IntegratorType, cone: ConeType, jacobian: JacobianType, solver: SolverType, iterations: int, ls_iterations: int, noslip_iterations: int, ccd_iterations: int, disableflags: DisableBit, enableflags: int, disableactuator: int, sdf_initpoints: int, sdf_iterations: int) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class SensorType(enum.IntEnum):
    """
    Type of sensor.
    
      Members:
        MAGNETOMETER: magnetometer
        CAMPROJECTION: camera projection
        RANGEFINDER: rangefinder
        JOINTPOS: joint position
        TENDONPOS: scalar tendon position
        ACTUATORPOS: actuator position
        BALLQUAT: ball joint orientation
        FRAMEPOS: frame position
        FRAMEXAXIS: frame x-axis
        FRAMEYAXIS: frame y-axis
        FRAMEZAXIS: frame z-axis
        FRAMEQUAT: frame orientation, represented as quaternion
        SUBTREECOM: subtree centor of mass
        CLOCK: simulation time
        VELOCIMETER: 3D linear velocity, in local frame
        GYRO: 3D angular velocity, in local frame
        JOINTVEL: joint velocity
        TENDONVEL: scalar tendon velocity
        ACTUATORVEL: actuator velocity
        BALLANGVEL: ball joint angular velocity
        FRAMELINVEL: 3D linear velocity
        FRAMEANGVEL: 3D angular velocity
        SUBTREELINVEL: subtree linear velocity
        SUBTREEANGMOM: subtree angular momentum
        TOUCH: scalar contact normal forces summed over the sensor zone
        ACCELEROMETER: accelerometer
        FORCE: force
        TORQUE: torque
        ACTUATORFRC: scalar actuator force
        JOINTACTFRC: scalar actuator force, measured at the joint
        FRAMELINACC: 3D linear acceleration
        FRAMEANGACC: 3D angular acceleration
      
    """
    ACCELEROMETER: typing.ClassVar[SensorType]  # value = <SensorType.ACCELEROMETER: 1>
    ACTUATORFRC: typing.ClassVar[SensorType]  # value = <SensorType.ACTUATORFRC: 15>
    ACTUATORPOS: typing.ClassVar[SensorType]  # value = <SensorType.ACTUATORPOS: 13>
    ACTUATORVEL: typing.ClassVar[SensorType]  # value = <SensorType.ACTUATORVEL: 14>
    BALLANGVEL: typing.ClassVar[SensorType]  # value = <SensorType.BALLANGVEL: 18>
    BALLQUAT: typing.ClassVar[SensorType]  # value = <SensorType.BALLQUAT: 17>
    CAMPROJECTION: typing.ClassVar[SensorType]  # value = <SensorType.CAMPROJECTION: 8>
    CLOCK: typing.ClassVar[SensorType]  # value = <SensorType.CLOCK: 42>
    FORCE: typing.ClassVar[SensorType]  # value = <SensorType.FORCE: 4>
    FRAMEANGACC: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEANGACC: 33>
    FRAMEANGVEL: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEANGVEL: 31>
    FRAMELINACC: typing.ClassVar[SensorType]  # value = <SensorType.FRAMELINACC: 32>
    FRAMELINVEL: typing.ClassVar[SensorType]  # value = <SensorType.FRAMELINVEL: 30>
    FRAMEPOS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEPOS: 25>
    FRAMEQUAT: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEQUAT: 26>
    FRAMEXAXIS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEXAXIS: 27>
    FRAMEYAXIS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEYAXIS: 28>
    FRAMEZAXIS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEZAXIS: 29>
    GYRO: typing.ClassVar[SensorType]  # value = <SensorType.GYRO: 3>
    JOINTACTFRC: typing.ClassVar[SensorType]  # value = <SensorType.JOINTACTFRC: 16>
    JOINTPOS: typing.ClassVar[SensorType]  # value = <SensorType.JOINTPOS: 9>
    JOINTVEL: typing.ClassVar[SensorType]  # value = <SensorType.JOINTVEL: 10>
    MAGNETOMETER: typing.ClassVar[SensorType]  # value = <SensorType.MAGNETOMETER: 6>
    RANGEFINDER: typing.ClassVar[SensorType]  # value = <SensorType.RANGEFINDER: 7>
    SUBTREEANGMOM: typing.ClassVar[SensorType]  # value = <SensorType.SUBTREEANGMOM: 36>
    SUBTREECOM: typing.ClassVar[SensorType]  # value = <SensorType.SUBTREECOM: 34>
    SUBTREELINVEL: typing.ClassVar[SensorType]  # value = <SensorType.SUBTREELINVEL: 35>
    TENDONPOS: typing.ClassVar[SensorType]  # value = <SensorType.TENDONPOS: 11>
    TENDONVEL: typing.ClassVar[SensorType]  # value = <SensorType.TENDONVEL: 12>
    TORQUE: typing.ClassVar[SensorType]  # value = <SensorType.TORQUE: 5>
    TOUCH: typing.ClassVar[SensorType]  # value = <SensorType.TOUCH: 0>
    VELOCIMETER: typing.ClassVar[SensorType]  # value = <SensorType.VELOCIMETER: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class SolverType(enum.IntEnum):
    """
    Constraint solver algorithm.
    
      Members:
        CG: Conjugate gradient (primal)
        NEWTON: Newton (primal)
      
    """
    CG: typing.ClassVar[SolverType]  # value = <SolverType.CG: 1>
    NEWTON: typing.ClassVar[SolverType]  # value = <SolverType.NEWTON: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class Statistic(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Model statistics (in qpos0).
    
      Attributes:
        meaninertia: mean diagonal inertia
        meanmass: mean body mass (not used)
        meansize: mean body size (not used)
        extent: spatial extent (not used)
        center: center of model (not used)
      
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'meaninertia': Field(name='meaninertia',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'meanmass': Field(name='meanmass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'meansize': Field(name='meansize',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'extent': Field(name='extent',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'center': Field(name='center',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True)
    __match_args__: typing.ClassVar[tuple] = ('meaninertia', 'meanmass', 'meansize', 'extent', 'center')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
            This is especially useful for frozen classes.  Example usage::
        
              @dataclass(frozen=True)
              class C:
                  x: int
                  y: int
        
              c = C(1, 2)
              c1 = replace(c, x=3)
              assert c1.x == 3 and c1.y == 2
            
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, meaninertia: jax.Array, meanmass: jax.Array, meansize: jax.Array, extent: jax.Array, center: jax.Array) -> None:
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class TrnType(enum.IntEnum):
    """
    Type of actuator transmission.
    
      Members:
        JOINT: force on joint
        JOINTINPARENT: force on joint, expressed in parent frame
        TENDON: force on tendon
        SITE: force on site
      
    """
    JOINT: typing.ClassVar[TrnType]  # value = <TrnType.JOINT: 0>
    JOINTINPARENT: typing.ClassVar[TrnType]  # value = <TrnType.JOINTINPARENT: 1>
    SITE: typing.ClassVar[TrnType]  # value = <TrnType.SITE: 4>
    TENDON: typing.ClassVar[TrnType]  # value = <TrnType.TENDON: 3>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
class WrapType(enum.IntEnum):
    """
    Type of tendon wrap object.
    
      Members:
        JOINT: constant moment arm
        PULLEY: pulley used to split tendon
        SITE: pass through site
        SPHERE: wrap around sphere
        CYLINDER: wrap around (infinite) cylinder
      
    """
    CYLINDER: typing.ClassVar[WrapType]  # value = <WrapType.CYLINDER: 5>
    JOINT: typing.ClassVar[WrapType]  # value = <WrapType.JOINT: 1>
    PULLEY: typing.ClassVar[WrapType]  # value = <WrapType.PULLEY: 2>
    SITE: typing.ClassVar[WrapType]  # value = <WrapType.SITE: 3>
    SPHERE: typing.ClassVar[WrapType]  # value = <WrapType.SPHERE: 4>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        ...
def _restricted_to(platform: str):
    """
    Specifies whether a field exists in only MuJoCo or MJX.
    """
