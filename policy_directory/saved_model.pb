��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758�
�
ActorNetwork/action/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameActorNetwork/action/bias
�
,ActorNetwork/action/bias/Read/ReadVariableOpReadVariableOpActorNetwork/action/bias*
_output_shapes
:*
dtype0
�
ActorNetwork/action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameActorNetwork/action/kernel
�
.ActorNetwork/action/kernel/Read/ReadVariableOpReadVariableOpActorNetwork/action/kernel*
_output_shapes
:	�*
dtype0
�
!ActorNetwork/input_mlp/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!ActorNetwork/input_mlp/dense/bias
�
5ActorNetwork/input_mlp/dense/bias/Read/ReadVariableOpReadVariableOp!ActorNetwork/input_mlp/dense/bias*
_output_shapes	
:�*
dtype0
�
#ActorNetwork/input_mlp/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#ActorNetwork/input_mlp/dense/kernel
�
7ActorNetwork/input_mlp/dense/kernel/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/kernel* 
_output_shapes
:
��*
dtype0
�
#ActorNetwork/input_mlp/dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#ActorNetwork/input_mlp/dense/bias_1
�
7ActorNetwork/input_mlp/dense/bias_1/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/bias_1*
_output_shapes	
:�*
dtype0
�
%ActorNetwork/input_mlp/dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	|�*6
shared_name'%ActorNetwork/input_mlp/dense/kernel_1
�
9ActorNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOpReadVariableOp%ActorNetwork/input_mlp/dense/kernel_1*
_output_shapes
:	|�*
dtype0
�
!ornstein_uhlenbeck_noise/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!ornstein_uhlenbeck_noise/Variable
�
5ornstein_uhlenbeck_noise/Variable/Read/ReadVariableOpReadVariableOp!ornstein_uhlenbeck_noise/Variable*
_output_shapes
:*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0_observationPlaceholder*'
_output_shapes
:���������|*
dtype0*
shape:���������|
j
action_0_rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0_step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/biasActorNetwork/action/kernelActorNetwork/action/bias!ornstein_uhlenbeck_noise/Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3564
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3576
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3598
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3591

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*
*
_wrapped_policy
_ou_process*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
get_metadata* 
ga
VARIABLE_VALUE!ornstein_uhlenbeck_noise/Variable,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE%ActorNetwork/input_mlp/dense/kernel_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/bias_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/kernel,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!ActorNetwork/input_mlp/dense/bias,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEActorNetwork/action/kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEActorNetwork/action/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*

_actor_network*

_x*
* 
* 
* 
* 
* 
* 
* 
* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_mlp_layers*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
 
)0
*1
+2
,3*
* 
 
)0
*1
+2
,3*
* 
* 
* 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

kernel
bias*
* 
* 
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable!ornstein_uhlenbeck_noise/Variable%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/biasActorNetwork/action/kernelActorNetwork/action/biasConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_3846
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable!ornstein_uhlenbeck_noise/Variable%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/biasActorNetwork/action/kernelActorNetwork/action/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_3880��
Y

__inference_<lambda>_614*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
,__inference_polymorphic_distribution_fn_3766
	step_type

reward
discount
observation��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� * 
fR
__inference__raise_3765*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������:���������:���������:���������|22
StatefulPartitionedCallStatefulPartitionedCall:TP
'
_output_shapes
:���������|
%
_user_specified_nameobservation:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:KG
#
_output_shapes
:���������
 
_user_specified_namereward:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type
�
$
"__inference_signature_wrapper_3598�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_function_with_signature_3594*(
_construction_contextkEagerRuntime*
_input_shapes 
�
h
(__inference_function_with_signature_3583
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_<lambda>_611^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�]
�
&__inference_polymorphic_action_fn_3523
	time_step
time_step_1
time_step_2
time_step_3N
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:	|�K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	�Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�E
2actornetwork_action_matmul_readvariableop_resource:	�A
3actornetwork_action_biasadd_readvariableop_resource:%
readvariableop_resource:
identity��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�AssignVariableOp�ReadVariableOp�add_1/ReadVariableOpk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����|   �
ActorNetwork/flatten/ReshapeReshapetime_step_3#ActorNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������|�
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	|�*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:�����������
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/mul/xConst*
_output_shapes
:*
dtype0*%
valueB"  �?  �?  �?   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/add/xConst*
_output_shapes
:*
dtype0*%
valueB"               �~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB l
Deterministic/sample/ShapeShapeActorNetwork/add:z:0*
T0*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToActorNetwork/add:z:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:����������
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:���������]
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes
:*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes
:o
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes
:b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��Y?W
mulMulmul/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
:M
addAddV2mul:z:0random_normal:z:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
add_1/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0�
add_1AddV2%Deterministic/sample/Reshape:output:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
clip_by_value/Minimum/yConst*
_output_shapes
:*
dtype0*%
valueB"  �?  �?  �?    
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������h
clip_by_value/yConst*
_output_shapes
:*
dtype0*%
valueB"  ��  ��  ��  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp^AssignVariableOp^ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������:���������:���������:���������|: : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:RN
'
_output_shapes
:���������|
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step
�
b
"__inference_signature_wrapper_3591
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_function_with_signature_3583^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
4
"__inference_signature_wrapper_3576

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_function_with_signature_3571*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
4
"__inference_get_initial_state_3570

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�H
�
__inference__traced_save_3846
file_prefix)
read_disablecopyonread_variable:	 H
:read_1_disablecopyonread_ornstein_uhlenbeck_noise_variable:Q
>read_2_disablecopyonread_actornetwork_input_mlp_dense_kernel_1:	|�K
<read_3_disablecopyonread_actornetwork_input_mlp_dense_bias_1:	�P
<read_4_disablecopyonread_actornetwork_input_mlp_dense_kernel:
��I
:read_5_disablecopyonread_actornetwork_input_mlp_dense_bias:	�F
3read_6_disablecopyonread_actornetwork_action_kernel:	�?
1read_7_disablecopyonread_actornetwork_action_bias:
savev2_const
identity_17��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: q
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_variable^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead:read_1_disablecopyonread_ornstein_uhlenbeck_noise_variable"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp:read_1_disablecopyonread_ornstein_uhlenbeck_noise_variable^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead>read_2_disablecopyonread_actornetwork_input_mlp_dense_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp>read_2_disablecopyonread_actornetwork_input_mlp_dense_kernel_1^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	|�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	|�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	|��
Read_3/DisableCopyOnReadDisableCopyOnRead<read_3_disablecopyonread_actornetwork_input_mlp_dense_bias_1"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp<read_3_disablecopyonread_actornetwork_input_mlp_dense_bias_1^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead<read_4_disablecopyonread_actornetwork_input_mlp_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp<read_4_disablecopyonread_actornetwork_input_mlp_dense_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_5/DisableCopyOnReadDisableCopyOnRead:read_5_disablecopyonread_actornetwork_input_mlp_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp:read_5_disablecopyonread_actornetwork_input_mlp_dense_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead3read_6_disablecopyonread_actornetwork_action_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp3read_6_disablecopyonread_actornetwork_action_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_7/DisableCopyOnReadDisableCopyOnRead1read_7_disablecopyonread_actornetwork_action_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp1read_7_disablecopyonread_actornetwork_action_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_16Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_17IdentityIdentity_16:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*'
_input_shapes
: : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp:	

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
,
__inference__raise_3765��Assert/Asserts
Assert/ConstConst*
_output_shapes
: *
dtype0*7
value.B, B&Distributions are not implemented yet.Y
Assert/Assert/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z {
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*7
value.B, B&Distributions are not implemented yet.z
Assert/AssertAssert Assert/Assert/condition:output:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes 2
Assert/AssertAssert/Assert
�
4
"__inference_get_initial_state_3769

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
*
(__inference_function_with_signature_3594�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_<lambda>_614*(
_construction_contextkEagerRuntime*
_input_shapes 
�(
�
 __inference__traced_restore_3880
file_prefix#
assignvariableop_variable:	 B
4assignvariableop_1_ornstein_uhlenbeck_noise_variable:K
8assignvariableop_2_actornetwork_input_mlp_dense_kernel_1:	|�E
6assignvariableop_3_actornetwork_input_mlp_dense_bias_1:	�J
6assignvariableop_4_actornetwork_input_mlp_dense_kernel:
��C
4assignvariableop_5_actornetwork_input_mlp_dense_bias:	�@
-assignvariableop_6_actornetwork_action_kernel:	�9
+assignvariableop_7_actornetwork_action_bias:

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp4assignvariableop_1_ornstein_uhlenbeck_noise_variableIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp8assignvariableop_2_actornetwork_input_mlp_dense_kernel_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_actornetwork_input_mlp_dense_bias_1Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_actornetwork_input_mlp_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp4assignvariableop_5_actornetwork_input_mlp_dense_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_actornetwork_action_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_actornetwork_action_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
:
(__inference_function_with_signature_3571

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_get_initial_state_3570*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�]
�
&__inference_polymorphic_action_fn_3676
	step_type

reward
discount
observationN
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:	|�K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	�Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�E
2actornetwork_action_matmul_readvariableop_resource:	�A
3actornetwork_action_biasadd_readvariableop_resource:%
readvariableop_resource:
identity��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�AssignVariableOp�ReadVariableOp�add_1/ReadVariableOpk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����|   �
ActorNetwork/flatten/ReshapeReshapeobservation#ActorNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������|�
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	|�*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:�����������
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/mul/xConst*
_output_shapes
:*
dtype0*%
valueB"  �?  �?  �?   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/add/xConst*
_output_shapes
:*
dtype0*%
valueB"               �~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB l
Deterministic/sample/ShapeShapeActorNetwork/add:z:0*
T0*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToActorNetwork/add:z:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:����������
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:���������]
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes
:*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes
:o
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes
:b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��Y?W
mulMulmul/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
:M
addAddV2mul:z:0random_normal:z:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
add_1/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0�
add_1AddV2%Deterministic/sample/Reshape:output:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
clip_by_value/Minimum/yConst*
_output_shapes
:*
dtype0*%
valueB"  �?  �?  �?    
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������h
clip_by_value/yConst*
_output_shapes
:*
dtype0*%
valueB"  ��  ��  ��  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp^AssignVariableOp^ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������:���������:���������:���������|: : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:TP
'
_output_shapes
:���������|
%
_user_specified_nameobservation:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:KG
#
_output_shapes
:���������
 
_user_specified_namereward:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type
�
�
"__inference_signature_wrapper_3564
discount
observation

reward
	step_type
unknown:	|�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_function_with_signature_3540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������:���������|:���������:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:VR
'
_output_shapes
:���������|
'
_user_specified_name0/observation:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount
�
_
__inference_<lambda>_611!
readvariableop_resource:	 
identity	��ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
�]
�
&__inference_polymorphic_action_fn_3754
time_step_step_type
time_step_reward
time_step_discount
time_step_observationN
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:	|�K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	�Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
��M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	�E
2actornetwork_action_matmul_readvariableop_resource:	�A
3actornetwork_action_biasadd_readvariableop_resource:%
readvariableop_resource:
identity��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp�AssignVariableOp�ReadVariableOp�add_1/ReadVariableOpk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����|   �
ActorNetwork/flatten/ReshapeReshapetime_step_observation#ActorNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������|�
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	|�*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:�����������
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/mul/xConst*
_output_shapes
:*
dtype0*%
valueB"  �?  �?  �?   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������k
ActorNetwork/add/xConst*
_output_shapes
:*
dtype0*%
valueB"               �~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB l
Deterministic/sample/ShapeShapeActorNetwork/add:z:0*
T0*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToActorNetwork/add:z:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:����������
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:���������]
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes
:*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes
:o
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes
:b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��Y?W
mulMulmul/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
:M
addAddV2mul:z:0random_normal:z:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
add_1/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0�
add_1AddV2%Deterministic/sample/Reshape:output:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
clip_by_value/Minimum/yConst*
_output_shapes
:*
dtype0*%
valueB"  �?  �?  �?    
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������h
clip_by_value/yConst*
_output_shapes
:*
dtype0*%
valueB"  ��  ��  ��  ��
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp^AssignVariableOp^ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������:���������:���������:���������|: : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
'
_output_shapes
:���������|
/
_user_specified_nametime_step_observation:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step_discount:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step_reward:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step_step_type
�
�
(__inference_function_with_signature_3540
	step_type

reward
discount
observation
unknown:	|�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_polymorphic_action_fn_3523o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������:���������:���������:���������|: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:VR
'
_output_shapes
:���������|
'
_user_specified_name0/observation:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type"�
L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0_discount:0���������
>
0/observation-
action_0_observation:0���������|
0
0/reward$
action_0_reward:0���������
6
0/step_type'
action_0_step_type:0���������:
action0
StatefulPartitionedCall:0���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:�[
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
R
0
1
2
3
4
5
6"
trackable_tuple_wrapper
F
_wrapped_policy
_ou_process"
trackable_dict_wrapper
�
trace_0
trace_12�
&__inference_polymorphic_action_fn_3676
&__inference_polymorphic_action_fn_3754�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_02�
,__inference_polymorphic_distribution_fn_3766�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
"__inference_get_initial_state_3769�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
__inference_<lambda>_614"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_<lambda>_611"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
-:+2!ornstein_uhlenbeck_noise/Variable
6:4	|�2#ActorNetwork/input_mlp/dense/kernel
0:.�2!ActorNetwork/input_mlp/dense/bias
7:5
��2#ActorNetwork/input_mlp/dense/kernel
0:.�2!ActorNetwork/input_mlp/dense/bias
-:+	�2ActorNetwork/action/kernel
&:$2ActorNetwork/action/bias
2
_actor_network"
_generic_user_object
&
_x"
_generic_user_object
�B�
&__inference_polymorphic_action_fn_3676	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_polymorphic_action_fn_3754time_step_step_typetime_step_rewardtime_step_discounttime_step_observation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_polymorphic_distribution_fn_3766	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_get_initial_state_3769
batch_size"�
���
FullArgSpec
args�
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3564
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3576
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3591"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3598"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_mlp_layers"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecE
args=�:
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�
� 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecE
args=�:
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�
� 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
<
)0
*1
+2
,3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
)0
*1
+2
,3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper@
__inference_<lambda>_611$�

� 
� "�
unknown 	0
__inference_<lambda>_614�

� 
� "� O
"__inference_get_initial_state_3769)"�
�
�

batch_size 
� "� �
&__inference_polymorphic_action_fn_3676����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������|
� 
� "V�S

PolicyStep*
action �
action���������
state� 
info� �
&__inference_polymorphic_action_fn_3754����
���
���
TimeStep6
	step_type)�&
time_step_step_type���������0
reward&�#
time_step_reward���������4
discount(�%
time_step_discount���������>
observation/�,
time_step_observation���������|
� 
� "V�S

PolicyStep*
action �
action���������
state� 
info� �
,__inference_polymorphic_distribution_fn_3766����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������|
� 
� "
 �
"__inference_signature_wrapper_3564����
� 
���
2
arg_0_discount �

0/discount���������
<
arg_0_observation'�$
0/observation���������|
.
arg_0_reward�
0/reward���������
4
arg_0_step_type!�
0/step_type���������"/�,
*
action �
action���������]
"__inference_signature_wrapper_357670�-
� 
&�#
!

batch_size�

batch_size "� V
"__inference_signature_wrapper_35910�

� 
� "�

int64�
int64 	:
"__inference_signature_wrapper_3598�

� 
� "� 