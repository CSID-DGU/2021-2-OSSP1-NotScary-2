/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Import webgl flags.
import './flags_webgl';
import { backend_util, buffer, DataStorage, engine, env, kernel_impls, KernelBackend, scalar, tidy, util } from '@tensorflow/tfjs-core';
import { getWebGLContext } from './canvas_util';
import { DecodeMatrixProgram } from './decode_matrix_gpu';
import { DecodeMatrixPackedProgram } from './decode_matrix_packed_gpu';
import { EncodeFloatProgram } from './encode_float_gpu';
import { EncodeFloatPackedProgram } from './encode_float_packed_gpu';
import { EncodeMatrixProgram } from './encode_matrix_gpu';
import { EncodeMatrixPackedProgram } from './encode_matrix_packed_gpu';
import { GPGPUContext } from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import { simpleAbsImplCPU } from './kernel_utils/shared';
import { PackProgram } from './pack_gpu';
import { ReshapePackedProgram } from './reshape_packed_gpu';
import * as tex_util from './tex_util';
import { TextureUsage } from './tex_util';
import { TextureManager } from './texture_manager';
import * as unary_op from './unaryop_gpu';
import { UnaryOpProgram } from './unaryop_gpu';
import { UnaryOpPackedProgram } from './unaryop_packed_gpu';
import { UnpackProgram } from './unpack_gpu';
import * as webgl_util from './webgl_util';
const whereImpl = kernel_impls.whereImpl;
export const EPSILON_FLOAT32 = 1e-7;
export const EPSILON_FLOAT16 = 1e-4;
const binaryCaches = {};
export function getBinaryCache(webGLVersion) {
    if (webGLVersion in binaryCaches) {
        return binaryCaches[webGLVersion];
    }
    binaryCaches[webGLVersion] = {};
    return binaryCaches[webGLVersion];
}
// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = env().getNumber('CPU_HANDOFF_SIZE_THRESHOLD');
// Empirically determined constant used to decide the number of MB on GPU
// before we warn about high memory use. The MB are this constant * screen area
// * dpi / 1024 / 1024.
const BEFORE_PAGING_CONSTANT = 600;
function numMBBeforeWarning() {
    if (env().global.screen == null) {
        return 1024; // 1 GB.
    }
    return (env().global.screen.height * env().global.screen.width *
        window.devicePixelRatio) *
        BEFORE_PAGING_CONSTANT / 1024 / 1024;
}
export class MathBackendWebGL extends KernelBackend {
    constructor(gpgpu) {
        super();
        // Maps data ids that have a pending read operation, to list of subscribers.
        this.pendingRead = new WeakMap();
        // List of data ids that are scheduled for disposal, but are waiting on a
        // pending read operation.
        this.pendingDisposal = new WeakSet();
        // Used to count the number of 'shallow' sliced tensors that point to the
        // same data id.
        this.dataRefCount = new WeakMap();
        this.numBytesInGPU = 0;
        // Accumulated time spent (including blocking) in uploading data to webgl.
        this.uploadWaitMs = 0;
        // Accumulated time spent (including blocking in downloading data from webgl.
        this.downloadWaitMs = 0;
        // record the last manual GL Flush time.
        this.lastGlFlushTime = 0;
        this.warnedAboutMemory = false;
        this.pendingDeletes = 0;
        this.disposed = false;
        if (!env().getBool('HAS_WEBGL')) {
            throw new Error('WebGL is not supported on this device');
        }
        if (gpgpu == null) {
            const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'));
            this.binaryCache = getBinaryCache(env().getNumber('WEBGL_VERSION'));
            this.gpgpu = new GPGPUContext(gl);
            this.canvas = gl.canvas;
            this.gpgpuCreatedLocally = true;
        }
        else {
            this.gpgpu = gpgpu;
            this.binaryCache = {};
            this.gpgpuCreatedLocally = false;
            this.canvas = gpgpu.gl.canvas;
        }
        this.textureManager = new TextureManager(this.gpgpu);
        this.numMBBeforeWarning = numMBBeforeWarning();
        this.texData = new DataStorage(this, engine());
    }
    nextDataId() {
        return MathBackendWebGL.nextDataId++;
    }
    numDataIds() {
        return this.texData.numDataIds() - this.pendingDeletes;
    }
    write(values, shape, dtype) {
        if (env().getBool('WEBGL_CHECK_NUMERICAL_PROBLEMS') ||
            env().getBool('DEBUG')) {
            this.checkNumericalProblems(values);
        }
        if (dtype === 'complex64' && values != null) {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        const dataId = { id: this.nextDataId() };
        this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD, refCount: 1 });
        return dataId;
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.texData.has(dataId)) {
            const tensorData = this.texData.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    /** Increase refCount of a `TextureData`. */
    incRef(dataId) {
        const texData = this.texData.get(dataId);
        texData.refCount++;
    }
    /** Decrease refCount of a `TextureData`. */
    decRef(dataId) {
        if (this.texData.has(dataId)) {
            const texData = this.texData.get(dataId);
            texData.refCount--;
        }
    }
    move(dataId, values, shape, dtype, refCount) {
        if (env().getBool('DEBUG')) {
            this.checkNumericalProblems(values);
        }
        if (dtype === 'complex64') {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD, refCount });
    }
    disposeIntermediateTensorInfo(tensorInfo) {
        this.disposeData(tensorInfo.dataId);
    }
    readSync(dataId) {
        const texData = this.texData.get(dataId);
        const { values, dtype, complexTensorInfos, slice, shape, isPacked } = texData;
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const data = this.readSync(res.dataId);
            this.disposeIntermediateTensorInfo(res);
            return data;
        }
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        if (dtype === 'string') {
            return values;
        }
        const shouldTimeProgram = this.activeTimers != null;
        let start;
        if (shouldTimeProgram) {
            start = util.now();
        }
        let result;
        if (dtype === 'complex64') {
            const realValues = this.readSync(complexTensorInfos.real.dataId);
            const imagValues = this.readSync(complexTensorInfos.imag.dataId);
            result = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else {
            result = this.getValuesFromTexture(dataId);
        }
        if (shouldTimeProgram) {
            this.downloadWaitMs += util.now() - start;
        }
        return this.convertAndCacheOnCPU(dataId, result);
    }
    async read(dataId) {
        if (this.pendingRead.has(dataId)) {
            const subscribers = this.pendingRead.get(dataId);
            return new Promise(resolve => subscribers.push(resolve));
        }
        const texData = this.texData.get(dataId);
        const { values, shape, slice, dtype, complexTensorInfos, isPacked } = texData;
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const data = this.read(res.dataId);
            this.disposeIntermediateTensorInfo(res);
            return data;
        }
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        if (env().getBool('DEBUG')) {
            // getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') caused a blocking GPU call.
            // For performance reason, only check it for debugging. In production,
            // it doesn't handle this use case anyway, so behavior is not changed.
            if (!env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
                env().getNumber('WEBGL_VERSION') === 2) {
                throw new Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
                    `WEBGL_VERSION=2 not yet supported.`);
            }
        }
        let buffer = null;
        let tmpDownloadTarget;
        if (dtype !== 'complex64' && env().get('WEBGL_BUFFER_SUPPORTED')) {
            // Possibly copy the texture into a buffer before inserting a fence.
            tmpDownloadTarget = this.decode(dataId);
            const tmpData = this.texData.get(tmpDownloadTarget.dataId);
            buffer = this.gpgpu.createBufferFromTexture(tmpData.texture, ...tex_util.getDenseTexShape(shape));
        }
        this.pendingRead.set(dataId, []);
        if (dtype !== 'complex64') {
            // Create a fence and wait for it to resolve.
            await this.gpgpu.createAndWaitForFence();
        }
        // Download the values from the GPU.
        let vals;
        if (dtype === 'complex64') {
            const ps = await Promise.all([
                this.read(complexTensorInfos.real.dataId),
                this.read(complexTensorInfos.imag.dataId)
            ]);
            const realValues = ps[0];
            const imagValues = ps[1];
            vals = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else if (buffer == null) {
            vals = this.getValuesFromTexture(dataId);
        }
        else {
            const size = util.sizeFromShape(shape);
            vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
        }
        if (tmpDownloadTarget != null) {
            this.disposeIntermediateTensorInfo(tmpDownloadTarget);
        }
        if (buffer != null) {
            const gl = this.gpgpu.gl;
            webgl_util.callAndCheck(gl, () => gl.deleteBuffer(buffer));
        }
        const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);
        const subscribers = this.pendingRead.get(dataId);
        this.pendingRead.delete(dataId);
        // Notify all pending reads.
        subscribers.forEach(resolve => resolve(dTypeVals));
        if (this.pendingDisposal.has(dataId)) {
            this.pendingDisposal.delete(dataId);
            if (this.disposeData(dataId)) {
                engine().removeDataId(dataId, this);
            }
            this.pendingDeletes--;
        }
        return dTypeVals;
    }
    bufferSync(t) {
        const data = this.readSync(t.dataId);
        let decodedData = data;
        if (t.dtype === 'string') {
            try {
                // Decode the bytes into string.
                decodedData = data.map(d => util.decodeString(d));
            }
            catch (_a) {
                throw new Error('Failed to decode encoded string bytes into utf-8');
            }
        }
        return buffer(t.shape, t.dtype, decodedData);
    }
    checkNumericalProblems(values) {
        if (values == null) {
            return;
        }
        for (let i = 0; i < values.length; i++) {
            const num = values[i];
            if (!webgl_util.canBeRepresented(num)) {
                if (env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
                    throw Error(`The value ${num} cannot be represented with your ` +
                        `current settings. Consider enabling float32 rendering: ` +
                        `'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);
                }
                throw Error(`The value ${num} cannot be represented on this device.`);
            }
        }
    }
    getValuesFromTexture(dataId) {
        const { shape, dtype, isPacked } = this.texData.get(dataId);
        const size = util.sizeFromShape(shape);
        if (env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
            const tmpTarget = this.decode(dataId);
            const tmpData = this.texData.get(tmpTarget.dataId);
            const vals = this.gpgpu
                .downloadMatrixFromPackedTexture(tmpData.texture, ...tex_util.getDenseTexShape(shape))
                .subarray(0, size);
            this.disposeIntermediateTensorInfo(tmpTarget);
            return vals;
        }
        const shouldUsePackedProgram = env().getBool('WEBGL_PACK') && isPacked === true;
        const outputShape = shouldUsePackedProgram ? webgl_util.getShapeAs3D(shape) : shape;
        const program = shouldUsePackedProgram ?
            new EncodeFloatPackedProgram(outputShape) :
            new EncodeFloatProgram(outputShape);
        const output = this.runWebGLProgram(program, [{ shape: outputShape, dtype, dataId }], 'float32');
        const tmpData = this.texData.get(output.dataId);
        const vals = this.gpgpu
            .downloadByteEncodedFloatMatrixFromOutputTexture(tmpData.texture, tmpData.texShape[0], tmpData.texShape[1])
            .subarray(0, size);
        this.disposeIntermediateTensorInfo(output);
        return vals;
    }
    timerAvailable() {
        return env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0;
    }
    async time(f) {
        const oldActiveTimers = this.activeTimers;
        const newActiveTimers = [];
        let outerMostTime = false;
        if (this.programTimersStack == null) {
            this.programTimersStack = newActiveTimers;
            outerMostTime = true;
        }
        else {
            this.activeTimers.push(newActiveTimers);
        }
        this.activeTimers = newActiveTimers;
        f();
        // needing to split these up because util.flatten only accepts certain types
        const flattenedActiveTimerQueries = util.flatten(this.activeTimers.map((d) => d.query))
            .filter(d => d != null);
        const flattenedActiveTimerNames = util.flatten(this.activeTimers.map((d) => d.name))
            .filter(d => d != null);
        this.activeTimers = oldActiveTimers;
        if (outerMostTime) {
            this.programTimersStack = null;
        }
        const res = {
            uploadWaitMs: this.uploadWaitMs,
            downloadWaitMs: this.downloadWaitMs,
            kernelMs: null,
            wallMs: null // will be filled by the engine
        };
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            const kernelMs = await Promise.all(flattenedActiveTimerQueries);
            res['kernelMs'] = util.sum(kernelMs);
            res['getExtraProfileInfo'] = () => kernelMs.map((d, i) => ({ name: flattenedActiveTimerNames[i], ms: d }))
                .map(d => `${d.name}: ${d.ms}`)
                .join(', ');
        }
        else {
            res['kernelMs'] = {
                error: 'WebGL query timers are not supported in this environment.'
            };
        }
        this.uploadWaitMs = 0;
        this.downloadWaitMs = 0;
        return res;
    }
    memory() {
        return {
            unreliable: false,
            numBytesInGPU: this.numBytesInGPU,
            numBytesInGPUAllocated: this.textureManager.numBytesAllocated,
            numBytesInGPUFree: this.textureManager.numBytesFree
        };
    }
    startTimer() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            return this.gpgpu.beginQuery();
        }
        return { startMs: util.now(), endMs: null };
    }
    endTimer(query) {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            this.gpgpu.endQuery();
            return query;
        }
        query.endMs = util.now();
        return query;
    }
    async getQueryTime(query) {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            return this.gpgpu.waitForQueryAndGetTime(query);
        }
        const timerQuery = query;
        return timerQuery.endMs - timerQuery.startMs;
    }
    /**
     * Decrease the RefCount on the dataId and dispose the memory if the dataId
     * has 0 refCount. If there are pending read on the data, the disposal would
     * added to the pending delete queue. Return true if the dataId is removed
     * from backend or the backend does not contain the dataId, false if the
     * dataId is not removed. Memory may or may not be released even when dataId
     * is removed, which also depends on dataRefCount, see `releaseGPU`.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        if (this.pendingDisposal.has(dataId)) {
            return false;
        }
        // No-op if already disposed.
        if (!this.texData.has(dataId)) {
            return true;
        }
        // if force flag is set, change refCount to 0, this would ensure disposal
        // when added to the pendingDisposal queue. Memory may or may not be
        // released, which also depends on dataRefCount, see `releaseGPU`.
        if (force) {
            this.texData.get(dataId).refCount = 0;
        }
        else {
            this.texData.get(dataId).refCount--;
        }
        if (!force && this.texData.get(dataId).refCount > 0) {
            return false;
        }
        if (this.pendingRead.has(dataId)) {
            this.pendingDisposal.add(dataId);
            this.pendingDeletes++;
            return false;
        }
        this.releaseGPUData(dataId);
        const { complexTensorInfos } = this.texData.get(dataId);
        if (complexTensorInfos != null) {
            this.disposeData(complexTensorInfos.real.dataId, force);
            this.disposeData(complexTensorInfos.imag.dataId, force);
        }
        this.texData.delete(dataId);
        return true;
    }
    releaseGPUData(dataId) {
        const { texture, dtype, texShape, usage, isPacked, slice } = this.texData.get(dataId);
        const key = slice && slice.origDataId || dataId;
        const refCount = this.dataRefCount.get(key);
        if (refCount > 1) {
            this.dataRefCount.set(key, refCount - 1);
        }
        else {
            this.dataRefCount.delete(key);
            if (texture != null) {
                this.numBytesInGPU -= this.computeBytes(texShape, dtype);
                this.textureManager.releaseTexture(texture, texShape, usage, isPacked);
            }
        }
        const texData = this.texData.get(dataId);
        texData.texture = null;
        texData.texShape = null;
        texData.isPacked = false;
        texData.slice = null;
    }
    getTexture(dataId) {
        this.uploadToGPU(dataId);
        return this.texData.get(dataId).texture;
    }
    /**
     * Returns internal information for the specific data bucket. Used in unit
     * tests.
     */
    getDataInfo(dataId) {
        return this.texData.get(dataId);
    }
    /*
    Tests whether all the inputs to an op are small and on the CPU. This heuristic
    determines when it would be faster to execute a kernel on the CPU. WebGL
    kernels opt into running this check and forwarding when appropriate.
    TODO(https://github.com/tensorflow/tfjs/issues/872): Develop a more
    sustainable strategy for optimizing backend execution of ops.
     */
    shouldExecuteOnCPU(inputs, sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD) {
        return env().getBool('WEBGL_CPU_FORWARD') &&
            inputs.every(input => this.texData.get(input.dataId).texture == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
    }
    getGPGPUContext() {
        return this.gpgpu;
    }
    where(condition) {
        backend_util.warn('tf.where() in webgl locks the UI thread. ' +
            'Call tf.whereAsync() instead');
        const condVals = condition.dataSync();
        return whereImpl(condition.shape, condVals);
    }
    packedUnaryOp(x, op, dtype) {
        const program = new UnaryOpPackedProgram(x.shape, op);
        const outInfo = this.compileAndRun(program, [x], dtype);
        return engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
    }
    // TODO(msoulanille) remove this once the backend has been modularized
    // a copy is needed here to break a circular dependency.
    // Also remove the op from unary_op.
    abs(x) {
        // TODO: handle cases when x is complex.
        if (this.shouldExecuteOnCPU([x]) && x.dtype !== 'complex64') {
            const outValues = simpleAbsImplCPU(this.texData.get(x.dataId).values);
            return this.makeOutput(x.shape, x.dtype, outValues);
        }
        if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
            return this.packedUnaryOp(x, unary_op.ABS, x.dtype);
        }
        const program = new UnaryOpProgram(x.shape, unary_op.ABS);
        const outInfo = this.compileAndRun(program, [x]);
        return engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
    }
    makeTensorInfo(shape, dtype, values) {
        let dataId;
        if (dtype === 'string' && values != null && values.length > 0 &&
            util.isString(values[0])) {
            const encodedValues = values.map(d => util.encodeString(d));
            dataId = this.write(encodedValues, shape, dtype);
        }
        else {
            dataId = this.write(values, shape, dtype);
        }
        this.texData.get(dataId).usage = null;
        return { dataId, shape, dtype };
    }
    makeOutput(shape, dtype, values) {
        const { dataId } = this.makeTensorInfo(shape, dtype, values);
        return engine().makeTensorFromDataId(dataId, shape, dtype, this);
    }
    unpackTensor(input) {
        const program = new UnpackProgram(input.shape);
        return this.runWebGLProgram(program, [input], input.dtype);
    }
    packTensor(input) {
        const program = new PackProgram(input.shape);
        const preventEagerUnpackingOutput = true;
        return this.runWebGLProgram(program, [input], input.dtype, null /* customUniformValues */, preventEagerUnpackingOutput);
    }
    packedReshape(input, afterShape) {
        const input3DShape = [
            webgl_util.getBatchDim(input.shape),
            ...webgl_util.getRowsCols(input.shape)
        ];
        const input3D = {
            dtype: input.dtype,
            shape: input3DShape,
            dataId: input.dataId
        };
        const afterShapeAs3D = [
            webgl_util.getBatchDim(afterShape), ...webgl_util.getRowsCols(afterShape)
        ];
        const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
        const preventEagerUnpackingOfOutput = true;
        const customValues = [input3DShape];
        const output = this.runWebGLProgram(program, [input3D], input.dtype, customValues, preventEagerUnpackingOfOutput);
        return { dataId: output.dataId, shape: afterShape, dtype: output.dtype };
    }
    decode(dataId) {
        const texData = this.texData.get(dataId);
        const { isPacked, shape, dtype } = texData;
        const shapeAs3D = webgl_util.getShapeAs3D(shape);
        let program;
        const denseTexShape = tex_util.getDenseTexShape(shapeAs3D);
        if (isPacked) {
            program = new DecodeMatrixPackedProgram(shapeAs3D);
        }
        else {
            program = new DecodeMatrixProgram(shapeAs3D);
        }
        const preventEagerUnpackingOfOutput = true;
        const customValues = [denseTexShape];
        const out = this.runWebGLProgram(program, [{ shape: shapeAs3D, dtype, dataId }], dtype, customValues, preventEagerUnpackingOfOutput);
        return { dtype, shape, dataId: out.dataId };
    }
    runWebGLProgram(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput = false) {
        const output = this.makeTensorInfo(program.outputShape, outputDtype);
        const outData = this.texData.get(output.dataId);
        if (program.packedOutput) {
            outData.isPacked = true;
        }
        if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
            const texelShape = tex_util.getDenseTexShape(program.outputShape);
            // For a densely packed output, we explicitly set texShape
            // so it doesn't get assigned later according to our typical packing
            // scheme wherein a single texel can only contain values from adjacent
            // rows/cols.
            outData.texShape = texelShape.map(d => d * 2);
        }
        if (program.outTexUsage != null) {
            outData.usage = program.outTexUsage;
        }
        if (util.sizeFromShape(output.shape) === 0) {
            // Short-circuit the computation since the result is empty (has 0 in its
            // shape).
            outData.values =
                util.getTypedArrayFromDType(output.dtype, 0);
            return output;
        }
        const dataToDispose = [];
        const inputsData = inputs.map(input => {
            if (input.dtype === 'complex64') {
                throw new Error(`GPGPUProgram does not support complex64 input. For complex64 ` +
                    `dtypes, please separate the program into real and imaginary ` +
                    `parts.`);
            }
            let texData = this.texData.get(input.dataId);
            if (texData.texture == null) {
                if (!program.packedInputs &&
                    util.sizeFromShape(input.shape) <=
                        env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
                    // Upload small tensors that live on the CPU as uniforms, not as
                    // textures. Do this only when the environment supports 32bit floats
                    // due to problems when comparing 16bit floats with 32bit floats.
                    // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
                    // possible for packed shaders to sample from uniforms.
                    return {
                        shape: input.shape,
                        texData: null,
                        isUniform: true,
                        uniformValues: texData.values
                    };
                }
                // This ensures that if a packed program's inputs have not yet been
                // uploaded to the GPU, they get uploaded as packed right off the bat.
                if (program.packedInputs) {
                    texData.isPacked = true;
                    texData.shape = input.shape;
                }
            }
            this.uploadToGPU(input.dataId);
            if (!!texData.isPacked !== !!program.packedInputs) {
                input = texData.isPacked ? this.unpackTensor(input) :
                    this.packTensor(input);
                dataToDispose.push(input);
                texData = this.texData.get(input.dataId);
            }
            else if (texData.isPacked &&
                !webgl_util.isReshapeFree(texData.shape, input.shape)) {
                // This is a special case where a texture exists for a tensor
                // but the shapes are incompatible (due to packing constraints) because
                // the tensor did not have a chance to go through the packed reshape
                // shader. This only happens when we reshape the *same* tensor to form
                // *distinct* inputs to an op, e.g. dotting a vector with itself. This
                // case will disappear once packed uploading is the default.
                const savedInput = input;
                const targetShape = input.shape;
                input.shape = texData.shape;
                input = this.packedReshape(input, targetShape);
                dataToDispose.push(input);
                texData = this.texData.get(input.dataId);
                savedInput.shape = targetShape;
            }
            return { shape: input.shape, texData, isUniform: false };
        });
        this.uploadToGPU(output.dataId);
        const outputData = { shape: output.shape, texData: outData, isUniform: false };
        const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
        const binary = this.getAndSaveBinary(key, () => {
            return gpgpu_math.compileProgram(this.gpgpu, program, inputsData, outputData);
        });
        const shouldTimeProgram = this.activeTimers != null;
        let query;
        if (shouldTimeProgram) {
            query = this.startTimer();
        }
        gpgpu_math.runProgram(this.gpgpu, binary, inputsData, outputData, customUniformValues);
        dataToDispose.forEach(info => this.disposeIntermediateTensorInfo(info));
        if (shouldTimeProgram) {
            query = this.endTimer(query);
            this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime(query) });
        }
        const glFlushThreshold = env().get('WEBGL_FLUSH_THRESHOLD');
        // Manually GL flush requested
        if (glFlushThreshold > 0) {
            const time = util.now();
            if ((time - this.lastGlFlushTime) > glFlushThreshold) {
                this.gpgpu.gl.flush();
                this.lastGlFlushTime = time;
            }
        }
        if (!env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
            preventEagerUnpackingOfOutput === false) {
            const unpacked = this.unpackTensor(output);
            this.disposeIntermediateTensorInfo(output);
            return unpacked;
        }
        return output;
    }
    compileAndRun(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput = false) {
        outputDtype = outputDtype || inputs[0].dtype;
        const outInfo = this.runWebGLProgram(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput);
        return outInfo;
    }
    getAndSaveBinary(key, getBinary) {
        if (!(key in this.binaryCache)) {
            this.binaryCache[key] = getBinary();
        }
        return this.binaryCache[key];
    }
    getTextureManager() {
        return this.textureManager;
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        // Avoid disposing the compiled webgl programs during unit testing because
        // it slows down test execution.
        if (!env().getBool('IS_TEST')) {
            const allKeys = Object.keys(this.binaryCache);
            allKeys.forEach(key => {
                this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
                delete this.binaryCache[key];
            });
        }
        this.textureManager.dispose();
        if (this.canvas != null &&
            (typeof (HTMLCanvasElement) !== 'undefined' &&
                this.canvas instanceof HTMLCanvasElement)) {
            this.canvas.remove();
        }
        else {
            this.canvas = null;
        }
        if (this.gpgpuCreatedLocally) {
            this.gpgpu.program = null;
            this.gpgpu.dispose();
        }
        this.disposed = true;
    }
    floatPrecision() {
        if (this.floatPrecisionValue == null) {
            this.floatPrecisionValue = tidy(() => {
                if (!env().get('WEBGL_RENDER_FLOAT32_ENABLED')) {
                    // Momentarily switching DEBUG flag to false so we don't throw an
                    // error trying to upload a small value.
                    const debugFlag = env().getBool('DEBUG');
                    env().set('DEBUG', false);
                    const underflowCheckValue = this.abs(scalar(1e-8)).dataSync()[0];
                    env().set('DEBUG', debugFlag);
                    if (underflowCheckValue > 0) {
                        return 32;
                    }
                }
                return 16;
            });
        }
        return this.floatPrecisionValue;
    }
    /** Returns the smallest representable number.  */
    epsilon() {
        return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
    }
    uploadToGPU(dataId) {
        const texData = this.texData.get(dataId);
        const { shape, dtype, values, texture, usage, isPacked } = texData;
        if (texture != null) {
            // Array is already on GPU. No-op.
            return;
        }
        const shouldTimeProgram = this.activeTimers != null;
        let start;
        if (shouldTimeProgram) {
            start = util.now();
        }
        let texShape = texData.texShape;
        if (texShape == null) {
            texShape = webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
            texData.texShape = texShape;
        }
        if (values != null) {
            const shapeAs3D = webgl_util.getShapeAs3D(shape);
            let program;
            let width = texShape[1], height = texShape[0];
            const isByteArray = values instanceof Uint8Array || values instanceof Uint8ClampedArray;
            // texture for float array is PhysicalTextureType.PACKED_2X2_FLOAT32, we
            // need to make sure the upload uses the same packed size
            if (isPacked || !isByteArray) {
                [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(texShape[0], texShape[1]);
            }
            if (isPacked) {
                program = new EncodeMatrixPackedProgram(shapeAs3D, isByteArray);
            }
            else {
                program = new EncodeMatrixProgram(shapeAs3D, isByteArray);
            }
            // TexShape for float array needs to be the original shape, which byte
            // array needs to be packed size. This allow the data upload shape to be
            // matched with texture creation logic.
            const tempDenseInputTexShape = isByteArray ? [height, width] : texShape;
            const tempDenseInputHandle = this.makeTensorInfo(tempDenseInputTexShape, dtype);
            const tempDenseInputTexData = this.texData.get(tempDenseInputHandle.dataId);
            if (isByteArray) {
                tempDenseInputTexData.usage = TextureUsage.PIXELS;
            }
            else {
                tempDenseInputTexData.usage = TextureUsage.UPLOAD;
            }
            tempDenseInputTexData.texShape = tempDenseInputTexShape;
            this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(tempDenseInputHandle.dataId), width, height, values);
            const customValues = [[height, width]];
            // We want the output to remain packed regardless of the value of
            // WEBGL_PACK.
            const preventEagerUnpacking = true;
            const encodedOutputTarget = this.runWebGLProgram(program, [tempDenseInputHandle], dtype, customValues, preventEagerUnpacking);
            // Have the original texture assume the identity of the encoded output.
            const outputTexData = this.texData.get(encodedOutputTarget.dataId);
            texData.texture = outputTexData.texture;
            texData.texShape = outputTexData.texShape;
            texData.isPacked = outputTexData.isPacked;
            texData.usage = outputTexData.usage;
            this.disposeIntermediateTensorInfo(tempDenseInputHandle);
            this.texData.delete(encodedOutputTarget.dataId);
            // Once uploaded, don't store the values on cpu.
            texData.values = null;
            if (shouldTimeProgram) {
                this.uploadWaitMs += util.now() - start;
            }
        }
        else {
            const newTexture = this.acquireTexture(texShape, usage, dtype, isPacked);
            texData.texture = newTexture;
        }
    }
    convertAndCacheOnCPU(dataId, float32Values) {
        const texData = this.texData.get(dataId);
        const { dtype } = texData;
        this.releaseGPUData(dataId);
        if (float32Values != null) {
            texData.values = float32ToTypedArray(float32Values, dtype);
        }
        return texData.values;
    }
    acquireTexture(texShape, texType, dtype, isPacked) {
        this.numBytesInGPU += this.computeBytes(texShape, dtype);
        if (!this.warnedAboutMemory &&
            this.numBytesInGPU > this.numMBBeforeWarning * 1024 * 1024) {
            const mb = (this.numBytesInGPU / 1024 / 1024).toFixed(2);
            this.warnedAboutMemory = true;
            console.warn(`High memory usage in GPU: ${mb} MB, ` +
                `most likely due to a memory leak`);
        }
        return this.textureManager.acquireTexture(texShape, texType, isPacked);
    }
    computeBytes(shape, dtype) {
        return shape[0] * shape[1] * util.bytesPerElement(dtype);
    }
}
MathBackendWebGL.nextDataId = 0;
function float32ToTypedArray(a, dtype) {
    if (dtype === 'float32' || dtype === 'complex64') {
        return a;
    }
    else if (dtype === 'int32' || dtype === 'bool') {
        const result = (dtype === 'int32') ? new Int32Array(a.length) :
            new Uint8Array(a.length);
        for (let i = 0; i < result.length; ++i) {
            result[i] = Math.round(a[i]);
        }
        return result;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF93ZWJnbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvYmFja2VuZF93ZWJnbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxzQkFBc0I7QUFDdEIsT0FBTyxlQUFlLENBQUM7QUFHdkIsT0FBTyxFQUFDLFlBQVksRUFBaUIsTUFBTSxFQUFVLFdBQVcsRUFBd0IsTUFBTSxFQUFFLEdBQUcsRUFBRSxZQUFZLEVBQUUsYUFBYSxFQUFxRCxNQUFNLEVBQXdELElBQUksRUFBMEIsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFcFQsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUM5QyxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUN4RCxPQUFPLEVBQUMseUJBQXlCLEVBQUMsTUFBTSw0QkFBNEIsQ0FBQztBQUNyRSxPQUFPLEVBQUMsa0JBQWtCLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUN0RCxPQUFPLEVBQUMsd0JBQXdCLEVBQUMsTUFBTSwyQkFBMkIsQ0FBQztBQUNuRSxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUN4RCxPQUFPLEVBQUMseUJBQXlCLEVBQUMsTUFBTSw0QkFBNEIsQ0FBQztBQUNyRSxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDN0MsT0FBTyxLQUFLLFVBQVUsTUFBTSxjQUFjLENBQUM7QUFFM0MsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFDdkQsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUN2QyxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUMxRCxPQUFPLEtBQUssUUFBUSxNQUFNLFlBQVksQ0FBQztB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QyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUM5QjtRQUNELE9BQU8sTUFBMkIsQ0FBQztLQUNwQztTQUFNO1FBQ0wsTUFBTSxJQUFJLEtBQUssQ0FBQyxpQkFBaUIsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUMzQztBQUNILENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8vIEltcG9ydCB3ZWJnbCBmbGFncy5cbmltcG9ydCAnLi9mbGFnc193ZWJnbCc7XG5cbmltcG9ydCAqIGFzIHRmIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgQmFja2VuZFZhbHVlcywgYnVmZmVyLCBEYXRhSWQsIERhdGFTdG9yYWdlLCBEYXRhVHlwZSwgRGF0YVZhbHVlcywgZW5naW5lLCBlbnYsIGtlcm5lbF9pbXBscywgS2VybmVsQmFja2VuZCwgTWVtb3J5SW5mbywgTnVtZXJpY0RhdGFUeXBlLCBSYW5rLCBSZWN1cnNpdmVBcnJheSwgc2NhbGFyLCBTaGFwZU1hcCwgVGVuc29yLCBUZW5zb3IyRCwgVGVuc29yQnVmZmVyLCBUZW5zb3JJbmZvLCB0aWR5LCBUaW1pbmdJbmZvLCBUeXBlZEFycmF5LCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldFdlYkdMQ29udGV4dH0gZnJvbSAnLi9jYW52YXNfdXRpbCc7XG5pbXBvcnQge0RlY29kZU1hdHJpeFByb2dyYW19IGZyb20gJy4vZGVjb2RlX21hdHJpeF9ncHUnO1xuaW1wb3J0IHtEZWNvZGVNYXRyaXhQYWNrZWRQcm9ncmFtfSBmcm9tICcuL2RlY29kZV9tYXRyaXhfcGFja2VkX2dwdSc7XG5pbXBvcnQge0VuY29kZUZsb2F0UHJvZ3JhbX0gZnJvbSAnLi9lbmNvZGVfZmxvYXRfZ3B1JztcbmltcG9ydCB7RW5jb2RlRmxvYXRQYWNrZWRQcm9ncmFtfSBmcm9tICcuL2VuY29kZV9mbG9hdF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7RW5jb2RlTWF0cml4UHJvZ3JhbX0gZnJvbSAnLi9lbmNvZGVfbWF0cml4X2dwdSc7XG5pbXBvcnQge0VuY29kZU1hdHJpeFBhY2tlZFByb2dyYW19IGZyb20gJy4vZW5jb2RlX21hdHJpeF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtHUEdQVUJpbmFyeSwgR1BHUFVQcm9ncmFtLCBUZW5zb3JEYXRhfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtzaW1wbGVBYnNJbXBsQ1BVfSBmcm9tICcuL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuaW1wb3J0IHtQYWNrUHJvZ3JhbX0gZnJvbSAnLi9wYWNrX2dwdSc7XG5pbXBvcnQge1Jlc2hhcGVQYWNrZWRQcm9ncmFtfSBmcm9tICcuL3Jlc2hhcGVfcGFja2VkX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXhfdXRpbCBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCB7VGV4dHVyZURhdGEsIFRleHR1cmVVc2FnZX0gZnJvbSAnLi90ZXhfdXRpbCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB1bmFyeV9vcCBmcm9tICcuL3VuYXJ5b3BfZ3B1JztcbmltcG9ydCB7VW5hcnlPcFByb2dyYW19IGZyb20gJy4vdW5hcnlvcF9ncHUnO1xuaW1wb3J0IHtVbmFyeU9wUGFja2VkUHJvZ3JhbX0gZnJvbSAnLi91bmFyeW9wX3BhY2tlZF9ncHUnO1xuaW1wb3J0IHtVbnBhY2tQcm9ncmFtfSBmcm9tICcuL3VucGFja19ncHUnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5jb25zdCB3aGVyZUltcGwgPSBrZXJuZWxfaW1wbHMud2hlcmVJbXBsO1xuXG5leHBvcnQgY29uc3QgRVBTSUxPTl9GTE9BVDMyID0gMWUtNztcbmV4cG9ydCBjb25zdCBFUFNJTE9OX0ZMT0FUMTYgPSAxZS00O1xuXG50eXBlIEtlcm5lbEluZm8gPSB7XG4gIG5hbWU6IHN0cmluZzsgcXVlcnk6IFByb21pc2U8bnVtYmVyPjtcbn07XG5cbmV4cG9ydCB0eXBlIFRpbWVyTm9kZSA9IFJlY3Vyc2l2ZUFycmF5PEtlcm5lbEluZm8+fEtlcm5lbEluZm87XG5leHBvcnQgaW50ZXJmYWNlIENQVVRpbWVyUXVlcnkge1xuICBzdGFydE1zOiBudW1iZXI7XG4gIGVuZE1zPzogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdMTWVtb3J5SW5mbyBleHRlbmRzIE1lbW9yeUluZm8ge1xuICBudW1CeXRlc0luR1BVOiBudW1iZXI7XG4gIC8vIFRyYWNrcyB0aGUgdG90YWwgbnVtYmVyIG9mIGJ5dGVzIGFsbG9jYXRlZCBvbiB0aGUgR1BVLCBhY2NvdW50aW5nIGZvciB0aGVcbiAgLy8gcGh5c2ljYWwgdGV4dHVyZSB0eXBlLlxuICBudW1CeXRlc0luR1BVQWxsb2NhdGVkOiBudW1iZXI7XG4gIC8vIFRyYWNrcyBieXRlIHNpemUgb2YgdGV4dHVyZXMgdGhhdCB3ZXJlIGNyZWF0ZWQgYW5kIHRoZW4gbWFkZSBhdmFpbGFibGUgZm9yXG4gIC8vIHJldXNlIChkaXNwb3NlZCkuXG4gIG51bUJ5dGVzSW5HUFVGcmVlOiBudW1iZXI7XG4gIHVucmVsaWFibGU6IGJvb2xlYW47XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xUaW1pbmdJbmZvIGV4dGVuZHMgVGltaW5nSW5mbyB7XG4gIHVwbG9hZFdhaXRNczogbnVtYmVyO1xuICBkb3dubG9hZFdhaXRNczogbnVtYmVyO1xufVxuXG5jb25zdCBiaW5hcnlDYWNoZXM6IHtbd2ViR0xWZXJzaW9uOiBzdHJpbmddOiB7W2tleTogc3RyaW5nXTogR1BHUFVCaW5hcnl9fSA9IHt9O1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0QmluYXJ5Q2FjaGUod2ViR0xWZXJzaW9uOiBudW1iZXIpIHtcbiAgaWYgKHdlYkdMVmVyc2lvbiBpbiBiaW5hcnlDYWNoZXMpIHtcbiAgICByZXR1cm4gYmluYXJ5Q2FjaGVzW3dlYkdMVmVyc2lvbl07XG4gIH1cbiAgYmluYXJ5Q2FjaGVzW3dlYkdMVmVyc2lvbl0gPSB7fTtcbiAgcmV0dXJuIGJpbmFyeUNhY2hlc1t3ZWJHTFZlcnNpb25dO1xufVxuXG4vLyBFbXBpcmljYWxseSBkZXRlcm1pbmVkIGNvbnN0YW50IHVzZWQgdG8gZGV0ZXJtaW5lIHNpemUgdGhyZXNob2xkIGZvciBoYW5kaW5nXG4vLyBvZmYgZXhlY3V0aW9uIHRvIHRoZSBDUFUuXG5jb25zdCBDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCA9XG4gICAgZW52KCkuZ2V0TnVtYmVyKCdDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCcpO1xuXG4vLyBFbXBpcmljYWxseSBkZXRlcm1pbmVkIGNvbnN0YW50IHVzZWQgdG8gZGVjaWRlIHRoZSBudW1iZXIgb2YgTUIgb24gR1BVXG4vLyBiZWZvcmUgd2Ugd2FybiBhYm91dCBoaWdoIG1lbW9yeSB1c2UuIFRoZSBNQiBhcmUgdGhpcyBjb25zdGFudCAqIHNjcmVlbiBhcmVhXG4vLyAqIGRwaSAvIDEwMjQgLyAxMDI0LlxuY29uc3QgQkVGT1JFX1BBR0lOR19DT05TVEFOVCA9IDYwMDtcbmZ1bmN0aW9uIG51bU1CQmVmb3JlV2FybmluZygpOiBudW1iZXIge1xuICBpZiAoZW52KCkuZ2xvYmFsLnNjcmVlbiA9PSBudWxsKSB7XG4gICAgcmV0dXJuIDEwMjQ7ICAvLyAxIEdCLlxuICB9XG4gIHJldHVybiAoZW52KCkuZ2xvYmFsLnNjcmVlbi5oZWlnaHQgKiBlbnYoKS5nbG9iYWwuc2NyZWVuLndpZHRoICpcbiAgICAgICAgICB3aW5kb3cuZGV2aWNlUGl4ZWxSYXRpbykgKlxuICAgICAgQkVGT1JFX1BBR0lOR19DT05TVEFOVCAvIDEwMjQgLyAxMDI0O1xufVxuXG5leHBvcnQgY2xhc3MgTWF0aEJhY2tlbmRXZWJHTCBleHRlbmRzIEtlcm5lbEJhY2tlbmQge1xuICB0ZXhEYXRhOiBEYXRhU3RvcmFnZTxUZXh0dXJlRGF0YT47XG4gIGdwZ3B1OiBHUEdQVUNvbnRleHQ7XG5cbiAgcHJpdmF0ZSBzdGF0aWMgbmV4dERhdGFJZCA9IDA7XG4gIHByaXZhdGUgbmV4dERhdGFJZCgpOiBudW1iZXIge1xuICAgIHJldHVybiBNYXRoQmFja2VuZFdlYkdMLm5leHREYXRhSWQrKztcbiAgfVxuICAvLyBNYXBzIGRhdGEgaWRzIHRoYXQgaGF2ZSBhIHBlbmRpbmcgcmVhZCBvcGVyYXRpb24sIHRvIGxpc3Qgb2Ygc3Vic2NyaWJlcnMuXG4gIHByaXZhdGUgcGVuZGluZ1JlYWQgPSBuZXcgV2Vha01hcDxEYXRhSWQsIEFycmF5PChhcnI6IFR5cGVkQXJyYXkpID0+IHZvaWQ+PigpO1xuICAvLyBMaXN0IG9mIGRhdGEgaWRzIHRoYXQgYXJlIHNjaGVkdWxlZCBmb3IgZGlzcG9zYWwsIGJ1dCBhcmUgd2FpdGluZyBvbiBhXG4gIC8vIHBlbmRpbmcgcmVhZCBvcGVyYXRpb24uXG4gIHByaXZhdGUgcGVuZGluZ0Rpc3Bvc2FsID0gbmV3IFdlYWtTZXQ8RGF0YUlkPigpO1xuXG4gIC8vIFVzZWQgdG8gY291bnQgdGhlIG51bWJlciBvZiAnc2hhbGxvdycgc2xpY2VkIHRlbnNvcnMgdGhhdCBwb2ludCB0byB0aGVcbiAgLy8gc2FtZSBkYXRhIGlkLlxuICBkYXRhUmVmQ291bnQgPSBuZXcgV2Vha01hcDxEYXRhSWQsIG51bWJlcj4oKTtcbiAgcHJpdmF0ZSBudW1CeXRlc0luR1BVID0gMDtcblxuICBwcml2YXRlIGNhbnZhczogSFRNTENhbnZhc0VsZW1lbnR8T2Zmc2NyZWVuQ2FudmFzO1xuXG4gIHByaXZhdGUgcHJvZ3JhbVRpbWVyc1N0YWNrOiBUaW1lck5vZGVbXTtcbiAgcHJpdmF0ZSBhY3RpdmVUaW1lcnM6IFRpbWVyTm9kZVtdO1xuICAvLyBBY2N1bXVsYXRlZCB0aW1lIHNwZW50IChpbmNsdWRpbmcgYmxvY2tpbmcpIGluIHVwbG9hZGluZyBkYXRhIHRvIHdlYmdsLlxuICBwcml2YXRlIHVwbG9hZFdhaXRNcyA9IDA7XG4gIC8vIEFjY3VtdWxhdGVkIHRpbWUgc3BlbnQgKGluY2x1ZGluZyBibG9ja2luZyBpbiBkb3dubG9hZGluZyBkYXRhIGZyb20gd2ViZ2wuXG4gIHByaXZhdGUgZG93bmxvYWRXYWl0TXMgPSAwO1xuXG4gIC8vIHJlY29yZCB0aGUgbGFzdCBtYW51YWwgR0wgRmx1c2ggdGltZS5cbiAgcHJpdmF0ZSBsYXN0R2xGbHVzaFRpbWUgPSAwO1xuXG4gIC8vIE51bWJlciBvZiBiaXRzIG9mIHByZWNpc2lvbiBvZiB0aGlzIGJhY2tlbmQuXG4gIHByaXZhdGUgZmxvYXRQcmVjaXNpb25WYWx1ZTogMzJ8MTY7XG5cbiAgcHJpdmF0ZSB0ZXh0dXJlTWFuYWdlcjogVGV4dHVyZU1hbmFnZXI7XG4gIHByaXZhdGUgYmluYXJ5Q2FjaGU6IHtba2V5OiBzdHJpbmddOiBHUEdQVUJpbmFyeX07XG4gIHByaXZhdGUgZ3BncHVDcmVhdGVkTG9jYWxseTogYm9vbGVhbjtcbiAgcHJpdmF0ZSBudW1NQkJlZm9yZVdhcm5pbmc6IG51bWJlcjtcbiAgcHJpdmF0ZSB3YXJuZWRBYm91dE1lbW9yeSA9IGZhbHNlO1xuXG4gIGNvbnN0cnVjdG9yKGdwZ3B1PzogR1BHUFVDb250ZXh0KSB7XG4gICAgc3VwZXIoKTtcbiAgICBpZiAoIWVudigpLmdldEJvb2woJ0hBU19XRUJHTCcpKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ1dlYkdMIGlzIG5vdCBzdXBwb3J0ZWQgb24gdGhpcyBkZXZpY2UnKTtcbiAgICB9XG5cbiAgICBpZiAoZ3BncHUgPT0gbnVsbCkge1xuICAgICAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9WRVJTSU9OJykpO1xuICAgICAgdGhpcy5iaW5hcnlDYWNoZSA9IGdldEJpbmFyeUNhY2hlKGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpKTtcbiAgICAgIHRoaXMuZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KGdsKTtcbiAgICAgIHRoaXMuY2FudmFzID0gZ2wuY2FudmFzO1xuICAgICAgdGhpcy5ncGdwdUNyZWF0ZWRMb2NhbGx5ID0gdHJ1ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5ncGdwdSA9IGdwZ3B1O1xuICAgICAgdGhpcy5iaW5hcnlDYWNoZSA9IHt9O1xuICAgICAgdGhpcy5ncGdwdUNyZWF0ZWRMb2NhbGx5ID0gZmFsc2U7XG4gICAgICB0aGlzLmNhbnZhcyA9IGdwZ3B1LmdsLmNhbnZhcztcbiAgICB9XG4gICAgdGhpcy50ZXh0dXJlTWFuYWdlciA9IG5ldyBUZXh0dXJlTWFuYWdlcih0aGlzLmdwZ3B1KTtcbiAgICB0aGlzLm51bU1CQmVmb3JlV2FybmluZyA9IG51bU1CQmVmb3JlV2FybmluZygpO1xuXG4gICAgdGhpcy50ZXhEYXRhID0gbmV3IERhdGFTdG9yYWdlKHRoaXMsIGVuZ2luZSgpKTtcbiAgfVxuXG4gIG51bURhdGFJZHMoKSB7XG4gICAgcmV0dXJuIHRoaXMudGV4RGF0YS5udW1EYXRhSWRzKCkgLSB0aGlzLnBlbmRpbmdEZWxldGVzO1xuICB9XG5cbiAgd3JpdGUodmFsdWVzOiBCYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSk6IERhdGFJZCB7XG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX0NIRUNLX05VTUVSSUNBTF9QUk9CTEVNUycpIHx8XG4gICAgICAgIGVudigpLmdldEJvb2woJ0RFQlVHJykpIHtcbiAgICAgIHRoaXMuY2hlY2tOdW1lcmljYWxQcm9ibGVtcyh2YWx1ZXMpO1xuICAgIH1cbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnICYmIHZhbHVlcyAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCB3cml0ZSB0byBhIGNvbXBsZXg2NCBkdHlwZS4gYCArXG4gICAgICAgICAgYFBsZWFzZSB1c2UgdGYuY29tcGxleChyZWFsLCBpbWFnKS5gKTtcbiAgICB9XG4gICAgY29uc3QgZGF0YUlkID0ge2lkOiB0aGlzLm5leHREYXRhSWQoKX07XG4gICAgdGhpcy50ZXhEYXRhLnNldChcbiAgICAgICAgZGF0YUlkLFxuICAgICAgICB7c2hhcGUsIGR0eXBlLCB2YWx1ZXMsIHVzYWdlOiBUZXh0dXJlVXNhZ2UuVVBMT0FELCByZWZDb3VudDogMX0pO1xuICAgIHJldHVybiBkYXRhSWQ7XG4gIH1cblxuICAvKiogUmV0dXJuIHJlZkNvdW50IG9mIGEgYFRlbnNvckRhdGFgLiAqL1xuICByZWZDb3VudChkYXRhSWQ6IERhdGFJZCk6IG51bWJlciB7XG4gICAgaWYgKHRoaXMudGV4RGF0YS5oYXMoZGF0YUlkKSkge1xuICAgICAgY29uc3QgdGVuc29yRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICAgIHJldHVybiB0ZW5zb3JEYXRhLnJlZkNvdW50O1xuICAgIH1cbiAgICByZXR1cm4gMDtcbiAgfVxuXG4gIC8qKiBJbmNyZWFzZSByZWZDb3VudCBvZiBhIGBUZXh0dXJlRGF0YWAuICovXG4gIGluY1JlZihkYXRhSWQ6IERhdGFJZCk6IHZvaWQge1xuICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgdGV4RGF0YS5yZWZDb3VudCsrO1xuICB9XG5cbiAgLyoqIERlY3JlYXNlIHJlZkNvdW50IG9mIGEgYFRleHR1cmVEYXRhYC4gKi9cbiAgZGVjUmVmKGRhdGFJZDogRGF0YUlkKTogdm9pZCB7XG4gICAgaWYgKHRoaXMudGV4RGF0YS5oYXMoZGF0YUlkKSkge1xuICAgICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICAgIHRleERhdGEucmVmQ291bnQtLTtcbiAgICB9XG4gIH1cblxuICBtb3ZlKFxuICAgICAgZGF0YUlkOiBEYXRhSWQsIHZhbHVlczogQmFja2VuZFZhbHVlcywgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgICByZWZDb3VudDogbnVtYmVyKTogdm9pZCB7XG4gICAgaWYgKGVudigpLmdldEJvb2woJ0RFQlVHJykpIHtcbiAgICAgIHRoaXMuY2hlY2tOdW1lcmljYWxQcm9ibGVtcyh2YWx1ZXMpO1xuICAgIH1cbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCB3cml0ZSB0byBhIGNvbXBsZXg2NCBkdHlwZS4gYCArXG4gICAgICAgICAgYFBsZWFzZSB1c2UgdGYuY29tcGxleChyZWFsLCBpbWFnKS5gKTtcbiAgICB9XG4gICAgdGhpcy50ZXhEYXRhLnNldChcbiAgICAgICAgZGF0YUlkLCB7c2hhcGUsIGR0eXBlLCB2YWx1ZXMsIHVzYWdlOiBUZXh0dXJlVXNhZ2UuVVBMT0FELCByZWZDb3VudH0pO1xuICB9XG5cbiAgZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8odGVuc29ySW5mbzogVGVuc29ySW5mbyk6IHZvaWQge1xuICAgIHRoaXMuZGlzcG9zZURhdGEodGVuc29ySW5mby5kYXRhSWQpO1xuICB9XG5cbiAgcmVhZFN5bmMoZGF0YUlkOiBEYXRhSWQpOiBCYWNrZW5kVmFsdWVzIHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHt2YWx1ZXMsIGR0eXBlLCBjb21wbGV4VGVuc29ySW5mb3MsIHNsaWNlLCBzaGFwZSwgaXNQYWNrZWR9ID0gdGV4RGF0YTtcblxuICAgIC8vIFRoZSBwcmVzZW5jZSBvZiBgc2xpY2VgIGluZGljYXRlcyB0aGlzIHRlbnNvciBpcyBhIHNoYWxsb3cgc2xpY2Ugb2YgYVxuICAgIC8vIGRpZmZlcmVudCB0ZW5zb3IsIGFuZCBpcyB1c2luZyB0aGF0IG9yaWdpbmFsIHRlbnNvcidzIHRleHR1cmUuIFJ1blxuICAgIC8vIGBjbG9uZWAgaW4gb3JkZXIgdG8gY29weSB0aGF0IHRleHR1cmUgYW5kIHJlYWQgZnJvbSBpdC5cbiAgICBpZiAoc2xpY2UgIT0gbnVsbCkge1xuICAgICAgbGV0IHByb2dyYW07XG4gICAgICBpZiAoaXNQYWNrZWQpIHtcbiAgICAgICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUGFja2VkUHJvZ3JhbShzaGFwZSwgdW5hcnlfb3AuQ0xPTkUpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUHJvZ3JhbShzaGFwZSwgdW5hcnlfb3AuQ0xPTkUpO1xuICAgICAgfVxuICAgICAgY29uc3QgcmVzID1cbiAgICAgICAgICB0aGlzLnJ1bldlYkdMUHJvZ3JhbShwcm9ncmFtLCBbe2RhdGFJZCwgc2hhcGUsIGR0eXBlfV0sIGR0eXBlKTtcbiAgICAgIGNvbnN0IGRhdGEgPSB0aGlzLnJlYWRTeW5jKHJlcy5kYXRhSWQpO1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXMpO1xuICAgICAgcmV0dXJuIGRhdGE7XG4gICAgfVxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkKTtcbiAgICB9XG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgcmV0dXJuIHZhbHVlcztcbiAgICB9XG4gICAgY29uc3Qgc2hvdWxkVGltZVByb2dyYW0gPSB0aGlzLmFjdGl2ZVRpbWVycyAhPSBudWxsO1xuICAgIGxldCBzdGFydDogbnVtYmVyO1xuICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSkge1xuICAgICAgc3RhcnQgPSB1dGlsLm5vdygpO1xuICAgIH1cblxuICAgIGxldCByZXN1bHQ6IEZsb2F0MzJBcnJheTtcbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICBjb25zdCByZWFsVmFsdWVzID1cbiAgICAgICAgICB0aGlzLnJlYWRTeW5jKGNvbXBsZXhUZW5zb3JJbmZvcy5yZWFsLmRhdGFJZCkgYXMgRmxvYXQzMkFycmF5O1xuICAgICAgY29uc3QgaW1hZ1ZhbHVlcyA9XG4gICAgICAgICAgdGhpcy5yZWFkU3luYyhjb21wbGV4VGVuc29ySW5mb3MuaW1hZy5kYXRhSWQpIGFzIEZsb2F0MzJBcnJheTtcbiAgICAgIHJlc3VsdCA9IGJhY2tlbmRfdXRpbC5tZXJnZVJlYWxBbmRJbWFnQXJyYXlzKHJlYWxWYWx1ZXMsIGltYWdWYWx1ZXMpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXN1bHQgPSB0aGlzLmdldFZhbHVlc0Zyb21UZXh0dXJlKGRhdGFJZCk7XG4gICAgfVxuXG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtKSB7XG4gICAgICB0aGlzLmRvd25sb2FkV2FpdE1zICs9IHV0aWwubm93KCkgLSBzdGFydDtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkLCByZXN1bHQpO1xuICB9XG5cbiAgYXN5bmMgcmVhZChkYXRhSWQ6IERhdGFJZCk6IFByb21pc2U8QmFja2VuZFZhbHVlcz4ge1xuICAgIGlmICh0aGlzLnBlbmRpbmdSZWFkLmhhcyhkYXRhSWQpKSB7XG4gICAgICBjb25zdCBzdWJzY3JpYmVycyA9IHRoaXMucGVuZGluZ1JlYWQuZ2V0KGRhdGFJZCk7XG4gICAgICByZXR1cm4gbmV3IFByb21pc2U8VHlwZWRBcnJheT4ocmVzb2x2ZSA9PiBzdWJzY3JpYmVycy5wdXNoKHJlc29sdmUpKTtcbiAgICB9XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7dmFsdWVzLCBzaGFwZSwgc2xpY2UsIGR0eXBlLCBjb21wbGV4VGVuc29ySW5mb3MsIGlzUGFja2VkfSA9IHRleERhdGE7XG5cbiAgICAvLyBUaGUgcHJlc2VuY2Ugb2YgYHNsaWNlYCBpbmRpY2F0ZXMgdGhpcyB0ZW5zb3IgaXMgYSBzaGFsbG93IHNsaWNlIG9mIGFcbiAgICAvLyBkaWZmZXJlbnQgdGVuc29yLCBhbmQgaXMgdXNpbmcgdGhhdCBvcmlnaW5hbCB0ZW5zb3IncyB0ZXh0dXJlLiBSdW5cbiAgICAvLyBgY2xvbmVgIGluIG9yZGVyIHRvIGNvcHkgdGhhdCB0ZXh0dXJlIGFuZCByZWFkIGZyb20gaXQuXG4gICAgaWYgKHNsaWNlICE9IG51bGwpIHtcbiAgICAgIGxldCBwcm9ncmFtO1xuICAgICAgaWYgKGlzUGFja2VkKSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFBhY2tlZFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHJlcyA9XG4gICAgICAgICAgdGhpcy5ydW5XZWJHTFByb2dyYW0ocHJvZ3JhbSwgW3tkYXRhSWQsIHNoYXBlLCBkdHlwZX1dLCBkdHlwZSk7XG4gICAgICBjb25zdCBkYXRhID0gdGhpcy5yZWFkKHJlcy5kYXRhSWQpO1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXMpO1xuICAgICAgcmV0dXJuIGRhdGE7XG4gICAgfVxuXG4gICAgaWYgKHZhbHVlcyAhPSBudWxsKSB7XG4gICAgICByZXR1cm4gdGhpcy5jb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQpO1xuICAgIH1cblxuICAgIGlmIChlbnYoKS5nZXRCb29sKCdERUJVRycpKSB7XG4gICAgICAvLyBnZXRCb29sKCdXRUJHTF9ET1dOTE9BRF9GTE9BVF9FTkFCTEVEJykgY2F1c2VkIGEgYmxvY2tpbmcgR1BVIGNhbGwuXG4gICAgICAvLyBGb3IgcGVyZm9ybWFuY2UgcmVhc29uLCBvbmx5IGNoZWNrIGl0IGZvciBkZWJ1Z2dpbmcuIEluIHByb2R1Y3Rpb24sXG4gICAgICAvLyBpdCBkb2Vzbid0IGhhbmRsZSB0aGlzIHVzZSBjYXNlIGFueXdheSwgc28gYmVoYXZpb3IgaXMgbm90IGNoYW5nZWQuXG4gICAgICBpZiAoIWVudigpLmdldEJvb2woJ1dFQkdMX0RPV05MT0FEX0ZMT0FUX0VOQUJMRUQnKSAmJlxuICAgICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpID09PSAyKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGB0ZW5zb3IuZGF0YSgpIHdpdGggV0VCR0xfRE9XTkxPQURfRkxPQVRfRU5BQkxFRD1mYWxzZSBhbmQgYCArXG4gICAgICAgICAgICBgV0VCR0xfVkVSU0lPTj0yIG5vdCB5ZXQgc3VwcG9ydGVkLmApO1xuICAgICAgfVxuICAgIH1cblxuICAgIGxldCBidWZmZXI6IFdlYkdMQnVmZmVyID0gbnVsbDtcbiAgICBsZXQgdG1wRG93bmxvYWRUYXJnZXQ6IFRlbnNvckluZm87XG5cbiAgICBpZiAoZHR5cGUgIT09ICdjb21wbGV4NjQnICYmIGVudigpLmdldCgnV0VCR0xfQlVGRkVSX1NVUFBPUlRFRCcpKSB7XG4gICAgICAvLyBQb3NzaWJseSBjb3B5IHRoZSB0ZXh0dXJlIGludG8gYSBidWZmZXIgYmVmb3JlIGluc2VydGluZyBhIGZlbmNlLlxuICAgICAgdG1wRG93bmxvYWRUYXJnZXQgPSB0aGlzLmRlY29kZShkYXRhSWQpO1xuICAgICAgY29uc3QgdG1wRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQodG1wRG93bmxvYWRUYXJnZXQuZGF0YUlkKTtcblxuICAgICAgYnVmZmVyID0gdGhpcy5ncGdwdS5jcmVhdGVCdWZmZXJGcm9tVGV4dHVyZShcbiAgICAgICAgICB0bXBEYXRhLnRleHR1cmUsIC4uLnRleF91dGlsLmdldERlbnNlVGV4U2hhcGUoc2hhcGUpKTtcbiAgICB9XG5cbiAgICB0aGlzLnBlbmRpbmdSZWFkLnNldChkYXRhSWQsIFtdKTtcblxuICAgIGlmIChkdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIC8vIENyZWF0ZSBhIGZlbmNlIGFuZCB3YWl0IGZvciBpdCB0byByZXNvbHZlLlxuICAgICAgYXdhaXQgdGhpcy5ncGdwdS5jcmVhdGVBbmRXYWl0Rm9yRmVuY2UoKTtcbiAgICB9XG5cbiAgICAvLyBEb3dubG9hZCB0aGUgdmFsdWVzIGZyb20gdGhlIEdQVS5cbiAgICBsZXQgdmFsczogRmxvYXQzMkFycmF5O1xuICAgIGlmIChkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIGNvbnN0IHBzID0gYXdhaXQgUHJvbWlzZS5hbGwoW1xuICAgICAgICB0aGlzLnJlYWQoY29tcGxleFRlbnNvckluZm9zLnJlYWwuZGF0YUlkKSxcbiAgICAgICAgdGhpcy5yZWFkKGNvbXBsZXhUZW5zb3JJbmZvcy5pbWFnLmRhdGFJZClcbiAgICAgIF0pO1xuXG4gICAgICBjb25zdCByZWFsVmFsdWVzID0gcHNbMF07XG4gICAgICBjb25zdCBpbWFnVmFsdWVzID0gcHNbMV07XG4gICAgICB2YWxzID0gYmFja2VuZF91dGlsLm1lcmdlUmVhbEFuZEltYWdBcnJheXMoXG4gICAgICAgICAgcmVhbFZhbHVlcyBhcyBGbG9hdDMyQXJyYXksIGltYWdWYWx1ZXMgYXMgRmxvYXQzMkFycmF5KTtcbiAgICB9IGVsc2UgaWYgKGJ1ZmZlciA9PSBudWxsKSB7XG4gICAgICB2YWxzID0gdGhpcy5nZXRWYWx1ZXNGcm9tVGV4dHVyZShkYXRhSWQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICAgIHZhbHMgPSB0aGlzLmdwZ3B1LmRvd25sb2FkRmxvYXQzMk1hdHJpeEZyb21CdWZmZXIoYnVmZmVyLCBzaXplKTtcbiAgICB9XG4gICAgaWYgKHRtcERvd25sb2FkVGFyZ2V0ICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8odG1wRG93bmxvYWRUYXJnZXQpO1xuICAgIH1cbiAgICBpZiAoYnVmZmVyICE9IG51bGwpIHtcbiAgICAgIGNvbnN0IGdsID0gdGhpcy5ncGdwdS5nbDtcbiAgICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZWxldGVCdWZmZXIoYnVmZmVyKSk7XG4gICAgfVxuICAgIGNvbnN0IGRUeXBlVmFscyA9IHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkLCB2YWxzKTtcblxuICAgIGNvbnN0IHN1YnNjcmliZXJzID0gdGhpcy5wZW5kaW5nUmVhZC5nZXQoZGF0YUlkKTtcbiAgICB0aGlzLnBlbmRpbmdSZWFkLmRlbGV0ZShkYXRhSWQpO1xuXG4gICAgLy8gTm90aWZ5IGFsbCBwZW5kaW5nIHJlYWRzLlxuICAgIHN1YnNjcmliZXJzLmZvckVhY2gocmVzb2x2ZSA9PiByZXNvbHZlKGRUeXBlVmFscykpO1xuICAgIGlmICh0aGlzLnBlbmRpbmdEaXNwb3NhbC5oYXMoZGF0YUlkKSkge1xuICAgICAgdGhpcy5wZW5kaW5nRGlzcG9zYWwuZGVsZXRlKGRhdGFJZCk7XG4gICAgICBpZiAodGhpcy5kaXNwb3NlRGF0YShkYXRhSWQpKSB7XG4gICAgICAgIGVuZ2luZSgpLnJlbW92ZURhdGFJZChkYXRhSWQsIHRoaXMpO1xuICAgICAgfVxuICAgICAgdGhpcy5wZW5kaW5nRGVsZXRlcy0tO1xuICAgIH1cbiAgICByZXR1cm4gZFR5cGVWYWxzO1xuICB9XG5cbiAgYnVmZmVyU3luYzxSIGV4dGVuZHMgUmFuaz4odDogVGVuc29ySW5mbyk6IFRlbnNvckJ1ZmZlcjxSPiB7XG4gICAgY29uc3QgZGF0YSA9IHRoaXMucmVhZFN5bmModC5kYXRhSWQpO1xuICAgIGxldCBkZWNvZGVkRGF0YSA9IGRhdGEgYXMgRGF0YVZhbHVlcztcbiAgICBpZiAodC5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIC8vIERlY29kZSB0aGUgYnl0ZXMgaW50byBzdHJpbmcuXG4gICAgICAgIGRlY29kZWREYXRhID0gKGRhdGEgYXMgVWludDhBcnJheVtdKS5tYXAoZCA9PiB1dGlsLmRlY29kZVN0cmluZyhkKSk7XG4gICAgICB9IGNhdGNoIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gZGVjb2RlIGVuY29kZWQgc3RyaW5nIGJ5dGVzIGludG8gdXRmLTgnKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGJ1ZmZlcih0LnNoYXBlIGFzIFNoYXBlTWFwW1JdLCB0LmR0eXBlLCBkZWNvZGVkRGF0YSkgYXNcbiAgICAgICAgVGVuc29yQnVmZmVyPFI+O1xuICB9XG5cbiAgcHJpdmF0ZSBjaGVja051bWVyaWNhbFByb2JsZW1zKHZhbHVlczogQmFja2VuZFZhbHVlcyk6IHZvaWQge1xuICAgIGlmICh2YWx1ZXMgPT0gbnVsbCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgbnVtID0gdmFsdWVzW2ldIGFzIG51bWJlcjtcbiAgICAgIGlmICghd2ViZ2xfdXRpbC5jYW5CZVJlcHJlc2VudGVkKG51bSkpIHtcbiAgICAgICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX1JFTkRFUl9GTE9BVDMyX0NBUEFCTEUnKSkge1xuICAgICAgICAgIHRocm93IEVycm9yKFxuICAgICAgICAgICAgICBgVGhlIHZhbHVlICR7bnVtfSBjYW5ub3QgYmUgcmVwcmVzZW50ZWQgd2l0aCB5b3VyIGAgK1xuICAgICAgICAgICAgICBgY3VycmVudCBzZXR0aW5ncy4gQ29uc2lkZXIgZW5hYmxpbmcgZmxvYXQzMiByZW5kZXJpbmc6IGAgK1xuICAgICAgICAgICAgICBgJ3RmLmVudigpLnNldCgnV0VCR0xfUkVOREVSX0ZMT0FUMzJfRU5BQkxFRCcsIHRydWUpOydgKTtcbiAgICAgICAgfVxuICAgICAgICB0aHJvdyBFcnJvcihgVGhlIHZhbHVlICR7bnVtfSBjYW5ub3QgYmUgcmVwcmVzZW50ZWQgb24gdGhpcyBkZXZpY2UuYCk7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRWYWx1ZXNGcm9tVGV4dHVyZShkYXRhSWQ6IERhdGFJZCk6IEZsb2F0MzJBcnJheSB7XG4gICAgY29uc3Qge3NoYXBlLCBkdHlwZSwgaXNQYWNrZWR9ID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIGlmIChlbnYoKS5nZXRCb29sKCdXRUJHTF9ET1dOTE9BRF9GTE9BVF9FTkFCTEVEJykpIHtcbiAgICAgIGNvbnN0IHRtcFRhcmdldCA9IHRoaXMuZGVjb2RlKGRhdGFJZCk7XG4gICAgICBjb25zdCB0bXBEYXRhID0gdGhpcy50ZXhEYXRhLmdldCh0bXBUYXJnZXQuZGF0YUlkKTtcbiAgICAgIGNvbnN0IHZhbHMgPSB0aGlzLmdwZ3B1XG4gICAgICAgICAgICAgICAgICAgICAgIC5kb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgdG1wRGF0YS50ZXh0dXJlLCAuLi50ZXhfdXRpbC5nZXREZW5zZVRleFNoYXBlKHNoYXBlKSlcbiAgICAgICAgICAgICAgICAgICAgICAgLnN1YmFycmF5KDAsIHNpemUpO1xuXG4gICAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKHRtcFRhcmdldCk7XG5cbiAgICAgIHJldHVybiB2YWxzO1xuICAgIH1cblxuICAgIGNvbnN0IHNob3VsZFVzZVBhY2tlZFByb2dyYW0gPVxuICAgICAgICBlbnYoKS5nZXRCb29sKCdXRUJHTF9QQUNLJykgJiYgaXNQYWNrZWQgPT09IHRydWU7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPVxuICAgICAgICBzaG91bGRVc2VQYWNrZWRQcm9ncmFtID8gd2ViZ2xfdXRpbC5nZXRTaGFwZUFzM0Qoc2hhcGUpIDogc2hhcGU7XG4gICAgY29uc3QgcHJvZ3JhbSA9IHNob3VsZFVzZVBhY2tlZFByb2dyYW0gP1xuICAgICAgICBuZXcgRW5jb2RlRmxvYXRQYWNrZWRQcm9ncmFtKG91dHB1dFNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSkgOlxuICAgICAgICBuZXcgRW5jb2RlRmxvYXRQcm9ncmFtKG91dHB1dFNoYXBlKTtcbiAgICBjb25zdCBvdXRwdXQgPSB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW3tzaGFwZTogb3V0cHV0U2hhcGUsIGR0eXBlLCBkYXRhSWR9XSwgJ2Zsb2F0MzInKTtcbiAgICBjb25zdCB0bXBEYXRhID0gdGhpcy50ZXhEYXRhLmdldChvdXRwdXQuZGF0YUlkKTtcbiAgICBjb25zdCB2YWxzID1cbiAgICAgICAgdGhpcy5ncGdwdVxuICAgICAgICAgICAgLmRvd25sb2FkQnl0ZUVuY29kZWRGbG9hdE1hdHJpeEZyb21PdXRwdXRUZXh0dXJlKFxuICAgICAgICAgICAgICAgIHRtcERhdGEudGV4dHVyZSwgdG1wRGF0YS50ZXhTaGFwZVswXSwgdG1wRGF0YS50ZXhTaGFwZVsxXSlcbiAgICAgICAgICAgIC5zdWJhcnJheSgwLCBzaXplKTtcbiAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKG91dHB1dCk7XG5cbiAgICByZXR1cm4gdmFscztcbiAgfVxuXG4gIHRpbWVyQXZhaWxhYmxlKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID4gMDtcbiAgfVxuXG4gIGFzeW5jIHRpbWUoZjogKCkgPT4gdm9pZCk6IFByb21pc2U8V2ViR0xUaW1pbmdJbmZvPiB7XG4gICAgY29uc3Qgb2xkQWN0aXZlVGltZXJzID0gdGhpcy5hY3RpdmVUaW1lcnM7XG4gICAgY29uc3QgbmV3QWN0aXZlVGltZXJzOiBUaW1lck5vZGVbXSA9IFtdO1xuXG4gICAgbGV0IG91dGVyTW9zdFRpbWUgPSBmYWxzZTtcbiAgICBpZiAodGhpcy5wcm9ncmFtVGltZXJzU3RhY2sgPT0gbnVsbCkge1xuICAgICAgdGhpcy5wcm9ncmFtVGltZXJzU3RhY2sgPSBuZXdBY3RpdmVUaW1lcnM7XG4gICAgICBvdXRlck1vc3RUaW1lID0gdHJ1ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5hY3RpdmVUaW1lcnMucHVzaChuZXdBY3RpdmVUaW1lcnMpO1xuICAgIH1cbiAgICB0aGlzLmFjdGl2ZVRpbWVycyA9IG5ld0FjdGl2ZVRpbWVycztcblxuICAgIGYoKTtcblxuICAgIC8vIG5lZWRpbmcgdG8gc3BsaXQgdGhlc2UgdXAgYmVjYXVzZSB1dGlsLmZsYXR0ZW4gb25seSBhY2NlcHRzIGNlcnRhaW4gdHlwZXNcbiAgICBjb25zdCBmbGF0dGVuZWRBY3RpdmVUaW1lclF1ZXJpZXMgPVxuICAgICAgICB1dGlsLmZsYXR0ZW4odGhpcy5hY3RpdmVUaW1lcnMubWFwKChkOiBLZXJuZWxJbmZvKSA9PiBkLnF1ZXJ5KSlcbiAgICAgICAgICAgIC5maWx0ZXIoZCA9PiBkICE9IG51bGwpO1xuICAgIGNvbnN0IGZsYXR0ZW5lZEFjdGl2ZVRpbWVyTmFtZXMgPVxuICAgICAgICB1dGlsLmZsYXR0ZW4odGhpcy5hY3RpdmVUaW1lcnMubWFwKChkOiBLZXJuZWxJbmZvKSA9PiBkLm5hbWUpKVxuICAgICAgICAgICAgLmZpbHRlcihkID0+IGQgIT0gbnVsbCk7XG5cbiAgICB0aGlzLmFjdGl2ZVRpbWVycyA9IG9sZEFjdGl2ZVRpbWVycztcblxuICAgIGlmIChvdXRlck1vc3RUaW1lKSB7XG4gICAgICB0aGlzLnByb2dyYW1UaW1lcnNTdGFjayA9IG51bGw7XG4gICAgfVxuXG4gICAgY29uc3QgcmVzOiBXZWJHTFRpbWluZ0luZm8gPSB7XG4gICAgICB1cGxvYWRXYWl0TXM6IHRoaXMudXBsb2FkV2FpdE1zLFxuICAgICAgZG93bmxvYWRXYWl0TXM6IHRoaXMuZG93bmxvYWRXYWl0TXMsXG4gICAgICBrZXJuZWxNczogbnVsbCxcbiAgICAgIHdhbGxNczogbnVsbCAgLy8gd2lsbCBiZSBmaWxsZWQgYnkgdGhlIGVuZ2luZVxuICAgIH07XG5cbiAgICBpZiAoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fUkVMSUFCTEUnKSA+IDApIHtcbiAgICAgIGNvbnN0IGtlcm5lbE1zID0gYXdhaXQgUHJvbWlzZS5hbGwoZmxhdHRlbmVkQWN0aXZlVGltZXJRdWVyaWVzKTtcblxuICAgICAgcmVzWydrZXJuZWxNcyddID0gdXRpbC5zdW0oa2VybmVsTXMpO1xuICAgICAgcmVzWydnZXRFeHRyYVByb2ZpbGVJbmZvJ10gPSAoKSA9PlxuICAgICAgICAgIGtlcm5lbE1zLm1hcCgoZCwgaSkgPT4gKHtuYW1lOiBmbGF0dGVuZWRBY3RpdmVUaW1lck5hbWVzW2ldLCBtczogZH0pKVxuICAgICAgICAgICAgICAubWFwKGQgPT4gYCR7ZC5uYW1lfTogJHtkLm1zfWApXG4gICAgICAgICAgICAgIC5qb2luKCcsICcpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXNbJ2tlcm5lbE1zJ10gPSB7XG4gICAgICAgIGVycm9yOiAnV2ViR0wgcXVlcnkgdGltZXJzIGFyZSBub3Qgc3VwcG9ydGVkIGluIHRoaXMgZW52aXJvbm1lbnQuJ1xuICAgICAgfTtcbiAgICB9XG5cbiAgICB0aGlzLnVwbG9hZFdhaXRNcyA9IDA7XG4gICAgdGhpcy5kb3dubG9hZFdhaXRNcyA9IDA7XG4gICAgcmV0dXJuIHJlcztcbiAgfVxuICBtZW1vcnkoKTogV2ViR0xNZW1vcnlJbmZvIHtcbiAgICByZXR1cm4ge1xuICAgICAgdW5yZWxpYWJsZTogZmFsc2UsXG4gICAgICBudW1CeXRlc0luR1BVOiB0aGlzLm51bUJ5dGVzSW5HUFUsXG4gICAgICBudW1CeXRlc0luR1BVQWxsb2NhdGVkOiB0aGlzLnRleHR1cmVNYW5hZ2VyLm51bUJ5dGVzQWxsb2NhdGVkLFxuICAgICAgbnVtQnl0ZXNJbkdQVUZyZWU6IHRoaXMudGV4dHVyZU1hbmFnZXIubnVtQnl0ZXNGcmVlXG4gICAgfSBhcyBXZWJHTE1lbW9yeUluZm87XG4gIH1cblxuICBwcml2YXRlIHN0YXJ0VGltZXIoKTogV2ViR0xRdWVyeXxDUFVUaW1lclF1ZXJ5IHtcbiAgICBpZiAoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fUkVMSUFCTEUnKSA+IDApIHtcbiAgICAgIHJldHVybiB0aGlzLmdwZ3B1LmJlZ2luUXVlcnkoKTtcbiAgICB9XG4gICAgcmV0dXJuIHtzdGFydE1zOiB1dGlsLm5vdygpLCBlbmRNczogbnVsbH07XG4gIH1cblxuICBwcml2YXRlIGVuZFRpbWVyKHF1ZXJ5OiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnkpOiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnkge1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID4gMCkge1xuICAgICAgdGhpcy5ncGdwdS5lbmRRdWVyeSgpO1xuICAgICAgcmV0dXJuIHF1ZXJ5O1xuICAgIH1cbiAgICAocXVlcnkgYXMgQ1BVVGltZXJRdWVyeSkuZW5kTXMgPSB1dGlsLm5vdygpO1xuICAgIHJldHVybiBxdWVyeTtcbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgZ2V0UXVlcnlUaW1lKHF1ZXJ5OiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnkpOiBQcm9taXNlPG51bWJlcj4ge1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID4gMCkge1xuICAgICAgcmV0dXJuIHRoaXMuZ3BncHUud2FpdEZvclF1ZXJ5QW5kR2V0VGltZShxdWVyeSBhcyBXZWJHTFF1ZXJ5KTtcbiAgICB9XG4gICAgY29uc3QgdGltZXJRdWVyeSA9IHF1ZXJ5IGFzIENQVVRpbWVyUXVlcnk7XG4gICAgcmV0dXJuIHRpbWVyUXVlcnkuZW5kTXMgLSB0aW1lclF1ZXJ5LnN0YXJ0TXM7XG4gIH1cblxuICBwcml2YXRlIHBlbmRpbmdEZWxldGVzID0gMDtcblxuICAvKipcbiAgICogRGVjcmVhc2UgdGhlIFJlZkNvdW50IG9uIHRoZSBkYXRhSWQgYW5kIGRpc3Bvc2UgdGhlIG1lbW9yeSBpZiB0aGUgZGF0YUlkXG4gICAqIGhhcyAwIHJlZkNvdW50LiBJZiB0aGVyZSBhcmUgcGVuZGluZyByZWFkIG9uIHRoZSBkYXRhLCB0aGUgZGlzcG9zYWwgd291bGRcbiAgICogYWRkZWQgdG8gdGhlIHBlbmRpbmcgZGVsZXRlIHF1ZXVlLiBSZXR1cm4gdHJ1ZSBpZiB0aGUgZGF0YUlkIGlzIHJlbW92ZWRcbiAgICogZnJvbSBiYWNrZW5kIG9yIHRoZSBiYWNrZW5kIGRvZXMgbm90IGNvbnRhaW4gdGhlIGRhdGFJZCwgZmFsc2UgaWYgdGhlXG4gICAqIGRhdGFJZCBpcyBub3QgcmVtb3ZlZC4gTWVtb3J5IG1heSBvciBtYXkgbm90IGJlIHJlbGVhc2VkIGV2ZW4gd2hlbiBkYXRhSWRcbiAgICogaXMgcmVtb3ZlZCwgd2hpY2ggYWxzbyBkZXBlbmRzIG9uIGRhdGFSZWZDb3VudCwgc2VlIGByZWxlYXNlR1BVYC5cbiAgICogQHBhcmFtIGRhdGFJZFxuICAgKiBAb2FyYW0gZm9yY2UgT3B0aW9uYWwsIHJlbW92ZSB0aGUgZGF0YSByZWdhcmRsZXNzIG9mIHJlZkNvdW50XG4gICAqL1xuICBkaXNwb3NlRGF0YShkYXRhSWQ6IERhdGFJZCwgZm9yY2UgPSBmYWxzZSk6IGJvb2xlYW4ge1xuICAgIGlmICh0aGlzLnBlbmRpbmdEaXNwb3NhbC5oYXMoZGF0YUlkKSkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIC8vIE5vLW9wIGlmIGFscmVhZHkgZGlzcG9zZWQuXG4gICAgaWYgKCF0aGlzLnRleERhdGEuaGFzKGRhdGFJZCkpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIGlmIGZvcmNlIGZsYWcgaXMgc2V0LCBjaGFuZ2UgcmVmQ291bnQgdG8gMCwgdGhpcyB3b3VsZCBlbnN1cmUgZGlzcG9zYWxcbiAgICAvLyB3aGVuIGFkZGVkIHRvIHRoZSBwZW5kaW5nRGlzcG9zYWwgcXVldWUuIE1lbW9yeSBtYXkgb3IgbWF5IG5vdCBiZVxuICAgIC8vIHJlbGVhc2VkLCB3aGljaCBhbHNvIGRlcGVuZHMgb24gZGF0YVJlZkNvdW50LCBzZWUgYHJlbGVhc2VHUFVgLlxuICAgIGlmIChmb3JjZSkge1xuICAgICAgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnJlZkNvdW50ID0gMDtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnJlZkNvdW50LS07XG4gICAgfVxuXG4gICAgaWYgKCFmb3JjZSAmJiB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCkucmVmQ291bnQgPiAwKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMucGVuZGluZ1JlYWQuaGFzKGRhdGFJZCkpIHtcbiAgICAgIHRoaXMucGVuZGluZ0Rpc3Bvc2FsLmFkZChkYXRhSWQpO1xuICAgICAgdGhpcy5wZW5kaW5nRGVsZXRlcysrO1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIHRoaXMucmVsZWFzZUdQVURhdGEoZGF0YUlkKTtcbiAgICBjb25zdCB7Y29tcGxleFRlbnNvckluZm9zfSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBpZiAoY29tcGxleFRlbnNvckluZm9zICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzcG9zZURhdGEoY29tcGxleFRlbnNvckluZm9zLnJlYWwuZGF0YUlkLCBmb3JjZSk7XG4gICAgICB0aGlzLmRpc3Bvc2VEYXRhKGNvbXBsZXhUZW5zb3JJbmZvcy5pbWFnLmRhdGFJZCwgZm9yY2UpO1xuICAgIH1cblxuICAgIHRoaXMudGV4RGF0YS5kZWxldGUoZGF0YUlkKTtcblxuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgcHJpdmF0ZSByZWxlYXNlR1BVRGF0YShkYXRhSWQ6IERhdGFJZCk6IHZvaWQge1xuICAgIGNvbnN0IHt0ZXh0dXJlLCBkdHlwZSwgdGV4U2hhcGUsIHVzYWdlLCBpc1BhY2tlZCwgc2xpY2V9ID1cbiAgICAgICAgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IGtleSA9IHNsaWNlICYmIHNsaWNlLm9yaWdEYXRhSWQgfHwgZGF0YUlkO1xuICAgIGNvbnN0IHJlZkNvdW50ID0gdGhpcy5kYXRhUmVmQ291bnQuZ2V0KGtleSk7XG5cbiAgICBpZiAocmVmQ291bnQgPiAxKSB7XG4gICAgICB0aGlzLmRhdGFSZWZDb3VudC5zZXQoa2V5LCByZWZDb3VudCAtIDEpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmRhdGFSZWZDb3VudC5kZWxldGUoa2V5KTtcbiAgICAgIGlmICh0ZXh0dXJlICE9IG51bGwpIHtcbiAgICAgICAgdGhpcy5udW1CeXRlc0luR1BVIC09IHRoaXMuY29tcHV0ZUJ5dGVzKHRleFNoYXBlLCBkdHlwZSk7XG4gICAgICAgIHRoaXMudGV4dHVyZU1hbmFnZXIucmVsZWFzZVRleHR1cmUodGV4dHVyZSwgdGV4U2hhcGUsIHVzYWdlLCBpc1BhY2tlZCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICB0ZXhEYXRhLnRleHR1cmUgPSBudWxsO1xuICAgIHRleERhdGEudGV4U2hhcGUgPSBudWxsO1xuICAgIHRleERhdGEuaXNQYWNrZWQgPSBmYWxzZTtcbiAgICB0ZXhEYXRhLnNsaWNlID0gbnVsbDtcbiAgfVxuXG4gIGdldFRleHR1cmUoZGF0YUlkOiBEYXRhSWQpOiBXZWJHTFRleHR1cmUge1xuICAgIHRoaXMudXBsb2FkVG9HUFUoZGF0YUlkKTtcbiAgICByZXR1cm4gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnRleHR1cmU7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBpbnRlcm5hbCBpbmZvcm1hdGlvbiBmb3IgdGhlIHNwZWNpZmljIGRhdGEgYnVja2V0LiBVc2VkIGluIHVuaXRcbiAgICogdGVzdHMuXG4gICAqL1xuICBnZXREYXRhSW5mbyhkYXRhSWQ6IERhdGFJZCk6IFRleHR1cmVEYXRhIHtcbiAgICByZXR1cm4gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICB9XG5cbiAgLypcbiAgVGVzdHMgd2hldGhlciBhbGwgdGhlIGlucHV0cyB0byBhbiBvcCBhcmUgc21hbGwgYW5kIG9uIHRoZSBDUFUuIFRoaXMgaGV1cmlzdGljXG4gIGRldGVybWluZXMgd2hlbiBpdCB3b3VsZCBiZSBmYXN0ZXIgdG8gZXhlY3V0ZSBhIGtlcm5lbCBvbiB0aGUgQ1BVLiBXZWJHTFxuICBrZXJuZWxzIG9wdCBpbnRvIHJ1bm5pbmcgdGhpcyBjaGVjayBhbmQgZm9yd2FyZGluZyB3aGVuIGFwcHJvcHJpYXRlLlxuICBUT0RPKGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzg3Mik6IERldmVsb3AgYSBtb3JlXG4gIHN1c3RhaW5hYmxlIHN0cmF0ZWd5IGZvciBvcHRpbWl6aW5nIGJhY2tlbmQgZXhlY3V0aW9uIG9mIG9wcy5cbiAgICovXG4gIHNob3VsZEV4ZWN1dGVPbkNQVShcbiAgICAgIGlucHV0czogVGVuc29ySW5mb1tdLFxuICAgICAgc2l6ZVRocmVzaG9sZCA9IENQVV9IQU5ET0ZGX1NJWkVfVEhSRVNIT0xEKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGVudigpLmdldEJvb2woJ1dFQkdMX0NQVV9GT1JXQVJEJykgJiZcbiAgICAgICAgaW5wdXRzLmV2ZXJ5KFxuICAgICAgICAgICAgaW5wdXQgPT4gdGhpcy50ZXhEYXRhLmdldChpbnB1dC5kYXRhSWQpLnRleHR1cmUgPT0gbnVsbCAmJlxuICAgICAgICAgICAgICAgIHV0aWwuc2l6ZUZyb21TaGFwZShpbnB1dC5zaGFwZSkgPCBzaXplVGhyZXNob2xkKTtcbiAgfVxuXG4gIGdldEdQR1BVQ29udGV4dCgpOiBHUEdQVUNvbnRleHQge1xuICAgIHJldHVybiB0aGlzLmdwZ3B1O1xuICB9XG5cbiAgd2hlcmUoY29uZGl0aW9uOiBUZW5zb3IpOiBUZW5zb3IyRCB7XG4gICAgYmFja2VuZF91dGlsLndhcm4oXG4gICAgICAgICd0Zi53aGVyZSgpIGluIHdlYmdsIGxvY2tzIHRoZSBVSSB0aHJlYWQuICcgK1xuICAgICAgICAnQ2FsbCB0Zi53aGVyZUFzeW5jKCkgaW5zdGVhZCcpO1xuICAgIGNvbnN0IGNvbmRWYWxzID0gY29uZGl0aW9uLmRhdGFTeW5jKCk7XG4gICAgcmV0dXJuIHdoZXJlSW1wbChjb25kaXRpb24uc2hhcGUsIGNvbmRWYWxzKTtcbiAgfVxuXG4gIHByaXZhdGUgcGFja2VkVW5hcnlPcCh4OiBUZW5zb3JJbmZvLCBvcDogc3RyaW5nLCBkdHlwZTogRGF0YVR5cGUpIHtcbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFVuYXJ5T3BQYWNrZWRQcm9ncmFtKHguc2hhcGUsIG9wKTtcbiAgICBjb25zdCBvdXRJbmZvID0gdGhpcy5jb21waWxlQW5kUnVuKHByb2dyYW0sIFt4XSwgZHR5cGUpO1xuICAgIHJldHVybiBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbURhdGFJZChcbiAgICAgICAgb3V0SW5mby5kYXRhSWQsIG91dEluZm8uc2hhcGUsIG91dEluZm8uZHR5cGUpO1xuICB9XG5cbiAgLy8gVE9ETyhtc291bGFuaWxsZSkgcmVtb3ZlIHRoaXMgb25jZSB0aGUgYmFja2VuZCBoYXMgYmVlbiBtb2R1bGFyaXplZFxuICAvLyBhIGNvcHkgaXMgbmVlZGVkIGhlcmUgdG8gYnJlYWsgYSBjaXJjdWxhciBkZXBlbmRlbmN5LlxuICAvLyBBbHNvIHJlbW92ZSB0aGUgb3AgZnJvbSB1bmFyeV9vcC5cbiAgYWJzPFQgZXh0ZW5kcyBUZW5zb3I+KHg6IFQpOiBUIHtcbiAgICAvLyBUT0RPOiBoYW5kbGUgY2FzZXMgd2hlbiB4IGlzIGNvbXBsZXguXG4gICAgaWYgKHRoaXMuc2hvdWxkRXhlY3V0ZU9uQ1BVKFt4XSkgJiYgeC5kdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIGNvbnN0IG91dFZhbHVlcyA9XG4gICAgICAgICAgc2ltcGxlQWJzSW1wbENQVSh0aGlzLnRleERhdGEuZ2V0KHguZGF0YUlkKS52YWx1ZXMgYXMgVHlwZWRBcnJheSk7XG4gICAgICByZXR1cm4gdGhpcy5tYWtlT3V0cHV0KHguc2hhcGUsIHguZHR5cGUsIG91dFZhbHVlcyk7XG4gICAgfVxuXG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX1BBQ0tfVU5BUllfT1BFUkFUSU9OUycpKSB7XG4gICAgICByZXR1cm4gdGhpcy5wYWNrZWRVbmFyeU9wKHgsIHVuYXJ5X29wLkFCUywgeC5kdHlwZSkgYXMgVDtcbiAgICB9XG5cbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFVuYXJ5T3BQcm9ncmFtKHguc2hhcGUsIHVuYXJ5X29wLkFCUyk7XG4gICAgY29uc3Qgb3V0SW5mbyA9IHRoaXMuY29tcGlsZUFuZFJ1bihwcm9ncmFtLCBbeF0pO1xuICAgIHJldHVybiBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbURhdGFJZChcbiAgICAgICAgICAgICAgIG91dEluZm8uZGF0YUlkLCBvdXRJbmZvLnNoYXBlLCBvdXRJbmZvLmR0eXBlKSBhcyBUO1xuICB9XG5cbiAgbWFrZVRlbnNvckluZm8oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIHZhbHVlcz86IEJhY2tlbmRWYWx1ZXN8c3RyaW5nW10pOiBUZW5zb3JJbmZvIHtcbiAgICBsZXQgZGF0YUlkO1xuICAgIGlmIChkdHlwZSA9PT0gJ3N0cmluZycgJiYgdmFsdWVzICE9IG51bGwgJiYgdmFsdWVzLmxlbmd0aCA+IDAgJiZcbiAgICAgICAgdXRpbC5pc1N0cmluZyh2YWx1ZXNbMF0pKSB7XG4gICAgICBjb25zdCBlbmNvZGVkVmFsdWVzID1cbiAgICAgICAgICAodmFsdWVzIGFzIHt9IGFzIHN0cmluZ1tdKS5tYXAoZCA9PiB1dGlsLmVuY29kZVN0cmluZyhkKSk7XG5cbiAgICAgIGRhdGFJZCA9IHRoaXMud3JpdGUoZW5jb2RlZFZhbHVlcywgc2hhcGUsIGR0eXBlKTtcbiAgICB9IGVsc2Uge1xuICAgICAgZGF0YUlkID0gdGhpcy53cml0ZSh2YWx1ZXMgYXMgVHlwZWRBcnJheSwgc2hhcGUsIGR0eXBlKTtcbiAgICB9XG5cbiAgICB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCkudXNhZ2UgPSBudWxsO1xuICAgIHJldHVybiB7ZGF0YUlkLCBzaGFwZSwgZHR5cGV9O1xuICB9XG5cbiAgcHJpdmF0ZSBtYWtlT3V0cHV0PFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgICAgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsIHZhbHVlcz86IEJhY2tlbmRWYWx1ZXMpOiBUIHtcbiAgICBjb25zdCB7ZGF0YUlkfSA9IHRoaXMubWFrZVRlbnNvckluZm8oc2hhcGUsIGR0eXBlLCB2YWx1ZXMpO1xuICAgIHJldHVybiBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbURhdGFJZChkYXRhSWQsIHNoYXBlLCBkdHlwZSwgdGhpcykgYXMgVDtcbiAgfVxuXG4gIHVucGFja1RlbnNvcihpbnB1dDogVGVuc29ySW5mbyk6IFRlbnNvckluZm8ge1xuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgVW5wYWNrUHJvZ3JhbShpbnB1dC5zaGFwZSk7XG4gICAgcmV0dXJuIHRoaXMucnVuV2ViR0xQcm9ncmFtKHByb2dyYW0sIFtpbnB1dF0sIGlucHV0LmR0eXBlKTtcbiAgfVxuXG4gIHBhY2tUZW5zb3IoaW5wdXQ6IFRlbnNvckluZm8pOiBUZW5zb3JJbmZvIHtcbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFBhY2tQcm9ncmFtKGlucHV0LnNoYXBlKTtcbiAgICBjb25zdCBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPdXRwdXQgPSB0cnVlO1xuICAgIHJldHVybiB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW2lucHV0XSwgaW5wdXQuZHR5cGUsIG51bGwgLyogY3VzdG9tVW5pZm9ybVZhbHVlcyAqLyxcbiAgICAgICAgcHJldmVudEVhZ2VyVW5wYWNraW5nT3V0cHV0KTtcbiAgfVxuXG4gIHByaXZhdGUgcGFja2VkUmVzaGFwZShpbnB1dDogVGVuc29ySW5mbywgYWZ0ZXJTaGFwZTogbnVtYmVyW10pOiBUZW5zb3JJbmZvIHtcbiAgICBjb25zdCBpbnB1dDNEU2hhcGUgPSBbXG4gICAgICB3ZWJnbF91dGlsLmdldEJhdGNoRGltKGlucHV0LnNoYXBlKSxcbiAgICAgIC4uLndlYmdsX3V0aWwuZ2V0Um93c0NvbHMoaW5wdXQuc2hhcGUpXG4gICAgXSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gICAgY29uc3QgaW5wdXQzRDogVGVuc29ySW5mbyA9IHtcbiAgICAgIGR0eXBlOiBpbnB1dC5kdHlwZSxcbiAgICAgIHNoYXBlOiBpbnB1dDNEU2hhcGUsXG4gICAgICBkYXRhSWQ6IGlucHV0LmRhdGFJZFxuICAgIH07XG4gICAgY29uc3QgYWZ0ZXJTaGFwZUFzM0QgPSBbXG4gICAgICB3ZWJnbF91dGlsLmdldEJhdGNoRGltKGFmdGVyU2hhcGUpLCAuLi53ZWJnbF91dGlsLmdldFJvd3NDb2xzKGFmdGVyU2hhcGUpXG4gICAgXSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG5cbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFJlc2hhcGVQYWNrZWRQcm9ncmFtKGFmdGVyU2hhcGVBczNELCBpbnB1dDNEU2hhcGUpO1xuICAgIGNvbnN0IHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0ID0gdHJ1ZTtcbiAgICBjb25zdCBjdXN0b21WYWx1ZXMgPSBbaW5wdXQzRFNoYXBlXTtcbiAgICBjb25zdCBvdXRwdXQgPSB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW2lucHV0M0RdLCBpbnB1dC5kdHlwZSwgY3VzdG9tVmFsdWVzLFxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCk7XG4gICAgcmV0dXJuIHtkYXRhSWQ6IG91dHB1dC5kYXRhSWQsIHNoYXBlOiBhZnRlclNoYXBlLCBkdHlwZTogb3V0cHV0LmR0eXBlfTtcbiAgfVxuXG4gIHByaXZhdGUgZGVjb2RlKGRhdGFJZDogRGF0YUlkKTogVGVuc29ySW5mbyB7XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7aXNQYWNrZWQsIHNoYXBlLCBkdHlwZX0gPSB0ZXhEYXRhO1xuICAgIGNvbnN0IHNoYXBlQXMzRCA9XG4gICAgICAgIHdlYmdsX3V0aWwuZ2V0U2hhcGVBczNEKHNoYXBlKSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gICAgbGV0IHByb2dyYW07XG4gICAgY29uc3QgZGVuc2VUZXhTaGFwZSA9IHRleF91dGlsLmdldERlbnNlVGV4U2hhcGUoc2hhcGVBczNEKTtcbiAgICBpZiAoaXNQYWNrZWQpIHtcbiAgICAgIHByb2dyYW0gPSBuZXcgRGVjb2RlTWF0cml4UGFja2VkUHJvZ3JhbShzaGFwZUFzM0QpO1xuICAgIH0gZWxzZSB7XG4gICAgICBwcm9ncmFtID0gbmV3IERlY29kZU1hdHJpeFByb2dyYW0oc2hhcGVBczNEKTtcbiAgICB9XG4gICAgY29uc3QgcHJldmVudEVhZ2VyVW5wYWNraW5nT2ZPdXRwdXQgPSB0cnVlO1xuICAgIGNvbnN0IGN1c3RvbVZhbHVlcyA9IFtkZW5zZVRleFNoYXBlXTtcbiAgICBjb25zdCBvdXQgPSB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW3tzaGFwZTogc2hhcGVBczNELCBkdHlwZSwgZGF0YUlkfV0sIGR0eXBlLCBjdXN0b21WYWx1ZXMsXG4gICAgICAgIHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0KTtcbiAgICByZXR1cm4ge2R0eXBlLCBzaGFwZSwgZGF0YUlkOiBvdXQuZGF0YUlkfTtcbiAgfVxuXG4gIHJ1bldlYkdMUHJvZ3JhbShcbiAgICAgIHByb2dyYW06IEdQR1BVUHJvZ3JhbSwgaW5wdXRzOiBUZW5zb3JJbmZvW10sIG91dHB1dER0eXBlOiBEYXRhVHlwZSxcbiAgICAgIGN1c3RvbVVuaWZvcm1WYWx1ZXM/OiBudW1iZXJbXVtdLFxuICAgICAgcHJldmVudEVhZ2VyVW5wYWNraW5nT2ZPdXRwdXQgPSBmYWxzZSk6IFRlbnNvckluZm8ge1xuICAgIGNvbnN0IG91dHB1dCA9IHRoaXMubWFrZVRlbnNvckluZm8ocHJvZ3JhbS5vdXRwdXRTaGFwZSwgb3V0cHV0RHR5cGUpO1xuICAgIGNvbnN0IG91dERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KG91dHB1dC5kYXRhSWQpO1xuICAgIGlmIChwcm9ncmFtLnBhY2tlZE91dHB1dCkge1xuICAgICAgb3V0RGF0YS5pc1BhY2tlZCA9IHRydWU7XG4gICAgfVxuICAgIGlmIChwcm9ncmFtLm91dFBhY2tpbmdTY2hlbWUgPT09IHRleF91dGlsLlBhY2tpbmdTY2hlbWUuREVOU0UpIHtcbiAgICAgIGNvbnN0IHRleGVsU2hhcGUgPSB0ZXhfdXRpbC5nZXREZW5zZVRleFNoYXBlKHByb2dyYW0ub3V0cHV0U2hhcGUpO1xuICAgICAgLy8gRm9yIGEgZGVuc2VseSBwYWNrZWQgb3V0cHV0LCB3ZSBleHBsaWNpdGx5IHNldCB0ZXhTaGFwZVxuICAgICAgLy8gc28gaXQgZG9lc24ndCBnZXQgYXNzaWduZWQgbGF0ZXIgYWNjb3JkaW5nIHRvIG91ciB0eXBpY2FsIHBhY2tpbmdcbiAgICAgIC8vIHNjaGVtZSB3aGVyZWluIGEgc2luZ2xlIHRleGVsIGNhbiBvbmx5IGNvbnRhaW4gdmFsdWVzIGZyb20gYWRqYWNlbnRcbiAgICAgIC8vIHJvd3MvY29scy5cbiAgICAgIG91dERhdGEudGV4U2hhcGUgPSB0ZXhlbFNoYXBlLm1hcChkID0+IGQgKiAyKSBhcyBbbnVtYmVyLCBudW1iZXJdO1xuICAgIH1cbiAgICBpZiAocHJvZ3JhbS5vdXRUZXhVc2FnZSAhPSBudWxsKSB7XG4gICAgICBvdXREYXRhLnVzYWdlID0gcHJvZ3JhbS5vdXRUZXhVc2FnZTtcbiAgICB9XG4gICAgaWYgKHV0aWwuc2l6ZUZyb21TaGFwZShvdXRwdXQuc2hhcGUpID09PSAwKSB7XG4gICAgICAvLyBTaG9ydC1jaXJjdWl0IHRoZSBjb21wdXRhdGlvbiBzaW5jZSB0aGUgcmVzdWx0IGlzIGVtcHR5IChoYXMgMCBpbiBpdHNcbiAgICAgIC8vIHNoYXBlKS5cbiAgICAgIG91dERhdGEudmFsdWVzID1cbiAgICAgICAgICB1dGlsLmdldFR5cGVkQXJyYXlGcm9tRFR5cGUob3V0cHV0LmR0eXBlIGFzICdmbG9hdDMyJywgMCk7XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICAgIH1cblxuICAgIGNvbnN0IGRhdGFUb0Rpc3Bvc2U6IFRlbnNvckluZm9bXSA9IFtdO1xuICAgIGNvbnN0IGlucHV0c0RhdGE6IFRlbnNvckRhdGFbXSA9IGlucHV0cy5tYXAoaW5wdXQgPT4ge1xuICAgICAgaWYgKGlucHV0LmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICBgR1BHUFVQcm9ncmFtIGRvZXMgbm90IHN1cHBvcnQgY29tcGxleDY0IGlucHV0LiBGb3IgY29tcGxleDY0IGAgK1xuICAgICAgICAgICAgYGR0eXBlcywgcGxlYXNlIHNlcGFyYXRlIHRoZSBwcm9ncmFtIGludG8gcmVhbCBhbmQgaW1hZ2luYXJ5IGAgK1xuICAgICAgICAgICAgYHBhcnRzLmApO1xuICAgICAgfVxuXG4gICAgICBsZXQgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoaW5wdXQuZGF0YUlkKTtcblxuICAgICAgaWYgKHRleERhdGEudGV4dHVyZSA9PSBudWxsKSB7XG4gICAgICAgIGlmICghcHJvZ3JhbS5wYWNrZWRJbnB1dHMgJiZcbiAgICAgICAgICAgIHV0aWwuc2l6ZUZyb21TaGFwZShpbnB1dC5zaGFwZSkgPD1cbiAgICAgICAgICAgICAgICBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1NJWkVfVVBMT0FEX1VOSUZPUk0nKSkge1xuICAgICAgICAgIC8vIFVwbG9hZCBzbWFsbCB0ZW5zb3JzIHRoYXQgbGl2ZSBvbiB0aGUgQ1BVIGFzIHVuaWZvcm1zLCBub3QgYXNcbiAgICAgICAgICAvLyB0ZXh0dXJlcy4gRG8gdGhpcyBvbmx5IHdoZW4gdGhlIGVudmlyb25tZW50IHN1cHBvcnRzIDMyYml0IGZsb2F0c1xuICAgICAgICAgIC8vIGR1ZSB0byBwcm9ibGVtcyB3aGVuIGNvbXBhcmluZyAxNmJpdCBmbG9hdHMgd2l0aCAzMmJpdCBmbG9hdHMuXG4gICAgICAgICAgLy8gVE9ETyhodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2lzc3Vlcy84MjEpOiBNYWtlIGl0XG4gICAgICAgICAgLy8gcG9zc2libGUgZm9yIHBhY2tlZCBzaGFkZXJzIHRvIHNhbXBsZSBmcm9tIHVuaWZvcm1zLlxuICAgICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICBzaGFwZTogaW5wdXQuc2hhcGUsXG4gICAgICAgICAgICB0ZXhEYXRhOiBudWxsLFxuICAgICAgICAgICAgaXNVbmlmb3JtOiB0cnVlLFxuICAgICAgICAgICAgdW5pZm9ybVZhbHVlczogdGV4RGF0YS52YWx1ZXMgYXMgVHlwZWRBcnJheVxuICAgICAgICAgIH07XG4gICAgICAgIH1cblxuICAgICAgICAvLyBUaGlzIGVuc3VyZXMgdGhhdCBpZiBhIHBhY2tlZCBwcm9ncmFtJ3MgaW5wdXRzIGhhdmUgbm90IHlldCBiZWVuXG4gICAgICAgIC8vIHVwbG9hZGVkIHRvIHRoZSBHUFUsIHRoZXkgZ2V0IHVwbG9hZGVkIGFzIHBhY2tlZCByaWdodCBvZmYgdGhlIGJhdC5cbiAgICAgICAgaWYgKHByb2dyYW0ucGFja2VkSW5wdXRzKSB7XG4gICAgICAgICAgdGV4RGF0YS5pc1BhY2tlZCA9IHRydWU7XG4gICAgICAgICAgdGV4RGF0YS5zaGFwZSA9IGlucHV0LnNoYXBlO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIHRoaXMudXBsb2FkVG9HUFUoaW5wdXQuZGF0YUlkKTtcbiAgICAgIGlmICghIXRleERhdGEuaXNQYWNrZWQgIT09ICEhcHJvZ3JhbS5wYWNrZWRJbnB1dHMpIHtcbiAgICAgICAgaW5wdXQgPSB0ZXhEYXRhLmlzUGFja2VkID8gdGhpcy51bnBhY2tUZW5zb3IoaW5wdXQpIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5wYWNrVGVuc29yKGlucHV0KTtcbiAgICAgICAgZGF0YVRvRGlzcG9zZS5wdXNoKGlucHV0KTtcbiAgICAgICAgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoaW5wdXQuZGF0YUlkKTtcbiAgICAgIH0gZWxzZSBpZiAoXG4gICAgICAgICAgdGV4RGF0YS5pc1BhY2tlZCAmJlxuICAgICAgICAgICF3ZWJnbF91dGlsLmlzUmVzaGFwZUZyZWUodGV4RGF0YS5zaGFwZSwgaW5wdXQuc2hhcGUpKSB7XG4gICAgICAgIC8vIFRoaXMgaXMgYSBzcGVjaWFsIGNhc2Ugd2hlcmUgYSB0ZXh0dXJlIGV4aXN0cyBmb3IgYSB0ZW5zb3JcbiAgICAgICAgLy8gYnV0IHRoZSBzaGFwZXMgYXJlIGluY29tcGF0aWJsZSAoZHVlIHRvIHBhY2tpbmcgY29uc3RyYWludHMpIGJlY2F1c2VcbiAgICAgICAgLy8gdGhlIHRlbnNvciBkaWQgbm90IGhhdmUgYSBjaGFuY2UgdG8gZ28gdGhyb3VnaCB0aGUgcGFja2VkIHJlc2hhcGVcbiAgICAgICAgLy8gc2hhZGVyLiBUaGlzIG9ubHkgaGFwcGVucyB3aGVuIHdlIHJlc2hhcGUgdGhlICpzYW1lKiB0ZW5zb3IgdG8gZm9ybVxuICAgICAgICAvLyAqZGlzdGluY3QqIGlucHV0cyB0byBhbiBvcCwgZS5nLiBkb3R0aW5nIGEgdmVjdG9yIHdpdGggaXRzZWxmLiBUaGlzXG4gICAgICAgIC8vIGNhc2Ugd2lsbCBkaXNhcHBlYXIgb25jZSBwYWNrZWQgdXBsb2FkaW5nIGlzIHRoZSBkZWZhdWx0LlxuXG4gICAgICAgIGNvbnN0IHNhdmVkSW5wdXQgPSBpbnB1dDtcbiAgICAgICAgY29uc3QgdGFyZ2V0U2hhcGUgPSBpbnB1dC5zaGFwZTtcblxuICAgICAgICBpbnB1dC5zaGFwZSA9IHRleERhdGEuc2hhcGU7XG4gICAgICAgIGlucHV0ID0gdGhpcy5wYWNrZWRSZXNoYXBlKGlucHV0IGFzIFRlbnNvciwgdGFyZ2V0U2hhcGUpO1xuICAgICAgICBkYXRhVG9EaXNwb3NlLnB1c2goaW5wdXQpO1xuICAgICAgICB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChpbnB1dC5kYXRhSWQpO1xuXG4gICAgICAgIHNhdmVkSW5wdXQuc2hhcGUgPSB0YXJnZXRTaGFwZTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIHtzaGFwZTogaW5wdXQuc2hhcGUsIHRleERhdGEsIGlzVW5pZm9ybTogZmFsc2V9O1xuICAgIH0pO1xuXG4gICAgdGhpcy51cGxvYWRUb0dQVShvdXRwdXQuZGF0YUlkKTtcbiAgICBjb25zdCBvdXRwdXREYXRhOlxuICAgICAgICBUZW5zb3JEYXRhID0ge3NoYXBlOiBvdXRwdXQuc2hhcGUsIHRleERhdGE6IG91dERhdGEsIGlzVW5pZm9ybTogZmFsc2V9O1xuICAgIGNvbnN0IGtleSA9IGdwZ3B1X21hdGgubWFrZVNoYWRlcktleShwcm9ncmFtLCBpbnB1dHNEYXRhLCBvdXRwdXREYXRhKTtcbiAgICBjb25zdCBiaW5hcnkgPSB0aGlzLmdldEFuZFNhdmVCaW5hcnkoa2V5LCAoKSA9PiB7XG4gICAgICByZXR1cm4gZ3BncHVfbWF0aC5jb21waWxlUHJvZ3JhbShcbiAgICAgICAgICB0aGlzLmdwZ3B1LCBwcm9ncmFtLCBpbnB1dHNEYXRhLCBvdXRwdXREYXRhKTtcbiAgICB9KTtcbiAgICBjb25zdCBzaG91bGRUaW1lUHJvZ3JhbSA9IHRoaXMuYWN0aXZlVGltZXJzICE9IG51bGw7XG4gICAgbGV0IHF1ZXJ5OiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnk7XG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtKSB7XG4gICAgICBxdWVyeSA9IHRoaXMuc3RhcnRUaW1lcigpO1xuICAgIH1cblxuICAgIGdwZ3B1X21hdGgucnVuUHJvZ3JhbShcbiAgICAgICAgdGhpcy5ncGdwdSwgYmluYXJ5LCBpbnB1dHNEYXRhLCBvdXRwdXREYXRhLCBjdXN0b21Vbmlmb3JtVmFsdWVzKTtcblxuICAgIGRhdGFUb0Rpc3Bvc2UuZm9yRWFjaChpbmZvID0+IHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8oaW5mbykpO1xuXG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtKSB7XG4gICAgICBxdWVyeSA9IHRoaXMuZW5kVGltZXIocXVlcnkpO1xuICAgICAgdGhpcy5hY3RpdmVUaW1lcnMucHVzaChcbiAgICAgICAgICB7bmFtZTogcHJvZ3JhbS5jb25zdHJ1Y3Rvci5uYW1lLCBxdWVyeTogdGhpcy5nZXRRdWVyeVRpbWUocXVlcnkpfSk7XG4gICAgfVxuXG4gICAgY29uc3QgZ2xGbHVzaFRocmVzaG9sZCA9IGVudigpLmdldCgnV0VCR0xfRkxVU0hfVEhSRVNIT0xEJyk7XG4gICAgLy8gTWFudWFsbHkgR0wgZmx1c2ggcmVxdWVzdGVkXG4gICAgaWYgKGdsRmx1c2hUaHJlc2hvbGQgPiAwKSB7XG4gICAgICBjb25zdCB0aW1lID0gdXRpbC5ub3coKTtcbiAgICAgIGlmICgodGltZSAtIHRoaXMubGFzdEdsRmx1c2hUaW1lKSA+IGdsRmx1c2hUaHJlc2hvbGQpIHtcbiAgICAgICAgdGhpcy5ncGdwdS5nbC5mbHVzaCgpO1xuICAgICAgICB0aGlzLmxhc3RHbEZsdXNoVGltZSA9IHRpbWU7XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdXRUJHTF9MQVpJTFlfVU5QQUNLJykgJiYgb3V0RGF0YS5pc1BhY2tlZCAmJlxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCA9PT0gZmFsc2UpIHtcbiAgICAgIGNvbnN0IHVucGFja2VkID0gdGhpcy51bnBhY2tUZW5zb3Iob3V0cHV0KTtcbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8ob3V0cHV0KTtcbiAgICAgIHJldHVybiB1bnBhY2tlZDtcbiAgICB9XG4gICAgcmV0dXJuIG91dHB1dDtcbiAgfVxuXG4gIGNvbXBpbGVBbmRSdW4oXG4gICAgICBwcm9ncmFtOiBHUEdQVVByb2dyYW0sIGlucHV0czogVGVuc29ySW5mb1tdLCBvdXRwdXREdHlwZT86IERhdGFUeXBlLFxuICAgICAgY3VzdG9tVW5pZm9ybVZhbHVlcz86IG51bWJlcltdW10sXG4gICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCA9IGZhbHNlKTogVGVuc29ySW5mbyB7XG4gICAgb3V0cHV0RHR5cGUgPSBvdXRwdXREdHlwZSB8fCBpbnB1dHNbMF0uZHR5cGU7XG4gICAgY29uc3Qgb3V0SW5mbyA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBpbnB1dHMsIG91dHB1dER0eXBlLCBjdXN0b21Vbmlmb3JtVmFsdWVzLFxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCk7XG4gICAgcmV0dXJuIG91dEluZm87XG4gIH1cblxuICBwcml2YXRlIGdldEFuZFNhdmVCaW5hcnkoa2V5OiBzdHJpbmcsIGdldEJpbmFyeTogKCkgPT4gR1BHUFVCaW5hcnkpOlxuICAgICAgR1BHUFVCaW5hcnkge1xuICAgIGlmICghKGtleSBpbiB0aGlzLmJpbmFyeUNhY2hlKSkge1xuICAgICAgdGhpcy5iaW5hcnlDYWNoZVtrZXldID0gZ2V0QmluYXJ5KCk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmJpbmFyeUNhY2hlW2tleV07XG4gIH1cblxuICBnZXRUZXh0dXJlTWFuYWdlcigpOiBUZXh0dXJlTWFuYWdlciB7XG4gICAgcmV0dXJuIHRoaXMudGV4dHVyZU1hbmFnZXI7XG4gIH1cblxuICBwcml2YXRlIGRpc3Bvc2VkID0gZmFsc2U7XG5cbiAgZGlzcG9zZSgpIHtcbiAgICBpZiAodGhpcy5kaXNwb3NlZCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICAvLyBBdm9pZCBkaXNwb3NpbmcgdGhlIGNvbXBpbGVkIHdlYmdsIHByb2dyYW1zIGR1cmluZyB1bml0IHRlc3RpbmcgYmVjYXVzZVxuICAgIC8vIGl0IHNsb3dzIGRvd24gdGVzdCBleGVjdXRpb24uXG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdJU19URVNUJykpIHtcbiAgICAgIGNvbnN0IGFsbEtleXMgPSBPYmplY3Qua2V5cyh0aGlzLmJpbmFyeUNhY2hlKTtcbiAgICAgIGFsbEtleXMuZm9yRWFjaChrZXkgPT4ge1xuICAgICAgICB0aGlzLmdwZ3B1LmRlbGV0ZVByb2dyYW0odGhpcy5iaW5hcnlDYWNoZVtrZXldLndlYkdMUHJvZ3JhbSk7XG4gICAgICAgIGRlbGV0ZSB0aGlzLmJpbmFyeUNhY2hlW2tleV07XG4gICAgICB9KTtcbiAgICB9XG4gICAgdGhpcy50ZXh0dXJlTWFuYWdlci5kaXNwb3NlKCk7XG4gICAgaWYgKHRoaXMuY2FudmFzICE9IG51bGwgJiZcbiAgICAgICAgKHR5cGVvZiAoSFRNTENhbnZhc0VsZW1lbnQpICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICAgICAgdGhpcy5jYW52YXMgaW5zdGFuY2VvZiBIVE1MQ2FudmFzRWxlbWVudCkpIHtcbiAgICAgIHRoaXMuY2FudmFzLnJlbW92ZSgpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmNhbnZhcyA9IG51bGw7XG4gICAgfVxuICAgIGlmICh0aGlzLmdwZ3B1Q3JlYXRlZExvY2FsbHkpIHtcbiAgICAgIHRoaXMuZ3BncHUucHJvZ3JhbSA9IG51bGw7XG4gICAgICB0aGlzLmdwZ3B1LmRpc3Bvc2UoKTtcbiAgICB9XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cblxuICBmbG9hdFByZWNpc2lvbigpOiAxNnwzMiB7XG4gICAgaWYgKHRoaXMuZmxvYXRQcmVjaXNpb25WYWx1ZSA9PSBudWxsKSB7XG4gICAgICB0aGlzLmZsb2F0UHJlY2lzaW9uVmFsdWUgPSB0aWR5KCgpID0+IHtcbiAgICAgICAgaWYgKCFlbnYoKS5nZXQoJ1dFQkdMX1JFTkRFUl9GTE9BVDMyX0VOQUJMRUQnKSkge1xuICAgICAgICAgIC8vIE1vbWVudGFyaWx5IHN3aXRjaGluZyBERUJVRyBmbGFnIHRvIGZhbHNlIHNvIHdlIGRvbid0IHRocm93IGFuXG4gICAgICAgICAgLy8gZXJyb3IgdHJ5aW5nIHRvIHVwbG9hZCBhIHNtYWxsIHZhbHVlLlxuICAgICAgICAgIGNvbnN0IGRlYnVnRmxhZyA9IGVudigpLmdldEJvb2woJ0RFQlVHJyk7XG4gICAgICAgICAgZW52KCkuc2V0KCdERUJVRycsIGZhbHNlKTtcbiAgICAgICAgICBjb25zdCB1bmRlcmZsb3dDaGVja1ZhbHVlID0gdGhpcy5hYnMoc2NhbGFyKDFlLTgpKS5kYXRhU3luYygpWzBdO1xuICAgICAgICAgIGVudigpLnNldCgnREVCVUcnLCBkZWJ1Z0ZsYWcpO1xuXG4gICAgICAgICAgaWYgKHVuZGVyZmxvd0NoZWNrVmFsdWUgPiAwKSB7XG4gICAgICAgICAgICByZXR1cm4gMzI7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHJldHVybiAxNjtcbiAgICAgIH0pO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5mbG9hdFByZWNpc2lvblZhbHVlO1xuICB9XG5cbiAgLyoqIFJldHVybnMgdGhlIHNtYWxsZXN0IHJlcHJlc2VudGFibGUgbnVtYmVyLiAgKi9cbiAgZXBzaWxvbigpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmZsb2F0UHJlY2lzaW9uKCkgPT09IDMyID8gRVBTSUxPTl9GTE9BVDMyIDogRVBTSUxPTl9GTE9BVDE2O1xuICB9XG5cbiAgdXBsb2FkVG9HUFUoZGF0YUlkOiBEYXRhSWQpOiB2b2lkIHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHtzaGFwZSwgZHR5cGUsIHZhbHVlcywgdGV4dHVyZSwgdXNhZ2UsIGlzUGFja2VkfSA9IHRleERhdGE7XG5cbiAgICBpZiAodGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICAvLyBBcnJheSBpcyBhbHJlYWR5IG9uIEdQVS4gTm8tb3AuXG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IHNob3VsZFRpbWVQcm9ncmFtID0gdGhpcy5hY3RpdmVUaW1lcnMgIT0gbnVsbDtcbiAgICBsZXQgc3RhcnQ6IG51bWJlcjtcbiAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgIHN0YXJ0ID0gdXRpbC5ub3coKTtcbiAgICB9XG5cbiAgICBsZXQgdGV4U2hhcGUgPSB0ZXhEYXRhLnRleFNoYXBlO1xuICAgIGlmICh0ZXhTaGFwZSA9PSBudWxsKSB7XG4gICAgICB0ZXhTaGFwZSA9IHdlYmdsX3V0aWwuZ2V0VGV4dHVyZVNoYXBlRnJvbUxvZ2ljYWxTaGFwZShzaGFwZSwgaXNQYWNrZWQpO1xuICAgICAgdGV4RGF0YS50ZXhTaGFwZSA9IHRleFNoYXBlO1xuICAgIH1cblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgY29uc3Qgc2hhcGVBczNEID0gd2ViZ2xfdXRpbC5nZXRTaGFwZUFzM0Qoc2hhcGUpO1xuXG4gICAgICBsZXQgcHJvZ3JhbTtcbiAgICAgIGxldCB3aWR0aCA9IHRleFNoYXBlWzFdLCBoZWlnaHQgPSB0ZXhTaGFwZVswXTtcbiAgICAgIGNvbnN0IGlzQnl0ZUFycmF5ID1cbiAgICAgICAgICB2YWx1ZXMgaW5zdGFuY2VvZiBVaW50OEFycmF5IHx8IHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4Q2xhbXBlZEFycmF5O1xuXG4gICAgICAvLyB0ZXh0dXJlIGZvciBmbG9hdCBhcnJheSBpcyBQaHlzaWNhbFRleHR1cmVUeXBlLlBBQ0tFRF8yWDJfRkxPQVQzMiwgd2VcbiAgICAgIC8vIG5lZWQgdG8gbWFrZSBzdXJlIHRoZSB1cGxvYWQgdXNlcyB0aGUgc2FtZSBwYWNrZWQgc2l6ZVxuICAgICAgaWYgKGlzUGFja2VkIHx8ICFpc0J5dGVBcnJheSkge1xuICAgICAgICBbd2lkdGgsIGhlaWdodF0gPSB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChcbiAgICAgICAgICAgIHRleFNoYXBlWzBdLCB0ZXhTaGFwZVsxXSk7XG4gICAgICB9XG5cbiAgICAgIGlmIChpc1BhY2tlZCkge1xuICAgICAgICBwcm9ncmFtID0gbmV3IEVuY29kZU1hdHJpeFBhY2tlZFByb2dyYW0oc2hhcGVBczNELCBpc0J5dGVBcnJheSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBwcm9ncmFtID0gbmV3IEVuY29kZU1hdHJpeFByb2dyYW0oc2hhcGVBczNELCBpc0J5dGVBcnJheSk7XG4gICAgICB9XG5cbiAgICAgIC8vIFRleFNoYXBlIGZvciBmbG9hdCBhcnJheSBuZWVkcyB0byBiZSB0aGUgb3JpZ2luYWwgc2hhcGUsIHdoaWNoIGJ5dGVcbiAgICAgIC8vIGFycmF5IG5lZWRzIHRvIGJlIHBhY2tlZCBzaXplLiBUaGlzIGFsbG93IHRoZSBkYXRhIHVwbG9hZCBzaGFwZSB0byBiZVxuICAgICAgLy8gbWF0Y2hlZCB3aXRoIHRleHR1cmUgY3JlYXRpb24gbG9naWMuXG4gICAgICBjb25zdCB0ZW1wRGVuc2VJbnB1dFRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICBpc0J5dGVBcnJheSA/IFtoZWlnaHQsIHdpZHRoXSA6IHRleFNoYXBlO1xuICAgICAgY29uc3QgdGVtcERlbnNlSW5wdXRIYW5kbGUgPVxuICAgICAgICAgIHRoaXMubWFrZVRlbnNvckluZm8odGVtcERlbnNlSW5wdXRUZXhTaGFwZSwgZHR5cGUpO1xuICAgICAgY29uc3QgdGVtcERlbnNlSW5wdXRUZXhEYXRhID1cbiAgICAgICAgICB0aGlzLnRleERhdGEuZ2V0KHRlbXBEZW5zZUlucHV0SGFuZGxlLmRhdGFJZCk7XG4gICAgICBpZiAoaXNCeXRlQXJyYXkpIHtcbiAgICAgICAgdGVtcERlbnNlSW5wdXRUZXhEYXRhLnVzYWdlID0gVGV4dHVyZVVzYWdlLlBJWEVMUztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRlbXBEZW5zZUlucHV0VGV4RGF0YS51c2FnZSA9IFRleHR1cmVVc2FnZS5VUExPQUQ7XG4gICAgICB9XG4gICAgICB0ZW1wRGVuc2VJbnB1dFRleERhdGEudGV4U2hhcGUgPSB0ZW1wRGVuc2VJbnB1dFRleFNoYXBlO1xuICAgICAgdGhpcy5ncGdwdS51cGxvYWREZW5zZU1hdHJpeFRvVGV4dHVyZShcbiAgICAgICAgICB0aGlzLmdldFRleHR1cmUodGVtcERlbnNlSW5wdXRIYW5kbGUuZGF0YUlkKSwgd2lkdGgsIGhlaWdodCxcbiAgICAgICAgICB2YWx1ZXMgYXMgVHlwZWRBcnJheSk7XG5cbiAgICAgIGNvbnN0IGN1c3RvbVZhbHVlcyA9IFtbaGVpZ2h0LCB3aWR0aF1dO1xuICAgICAgLy8gV2Ugd2FudCB0aGUgb3V0cHV0IHRvIHJlbWFpbiBwYWNrZWQgcmVnYXJkbGVzcyBvZiB0aGUgdmFsdWUgb2ZcbiAgICAgIC8vIFdFQkdMX1BBQ0suXG4gICAgICBjb25zdCBwcmV2ZW50RWFnZXJVbnBhY2tpbmcgPSB0cnVlO1xuICAgICAgY29uc3QgZW5jb2RlZE91dHB1dFRhcmdldCA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICAgIHByb2dyYW0sIFt0ZW1wRGVuc2VJbnB1dEhhbmRsZV0sIGR0eXBlLCBjdXN0b21WYWx1ZXMsXG4gICAgICAgICAgcHJldmVudEVhZ2VyVW5wYWNraW5nKTtcblxuICAgICAgLy8gSGF2ZSB0aGUgb3JpZ2luYWwgdGV4dHVyZSBhc3N1bWUgdGhlIGlkZW50aXR5IG9mIHRoZSBlbmNvZGVkIG91dHB1dC5cbiAgICAgIGNvbnN0IG91dHB1dFRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGVuY29kZWRPdXRwdXRUYXJnZXQuZGF0YUlkKTtcbiAgICAgIHRleERhdGEudGV4dHVyZSA9IG91dHB1dFRleERhdGEudGV4dHVyZTtcbiAgICAgIHRleERhdGEudGV4U2hhcGUgPSBvdXRwdXRUZXhEYXRhLnRleFNoYXBlO1xuICAgICAgdGV4RGF0YS5pc1BhY2tlZCA9IG91dHB1dFRleERhdGEuaXNQYWNrZWQ7XG4gICAgICB0ZXhEYXRhLnVzYWdlID0gb3V0cHV0VGV4RGF0YS51c2FnZTtcblxuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyh0ZW1wRGVuc2VJbnB1dEhhbmRsZSk7XG4gICAgICB0aGlzLnRleERhdGEuZGVsZXRlKGVuY29kZWRPdXRwdXRUYXJnZXQuZGF0YUlkKTtcblxuICAgICAgLy8gT25jZSB1cGxvYWRlZCwgZG9uJ3Qgc3RvcmUgdGhlIHZhbHVlcyBvbiBjcHUuXG4gICAgICB0ZXhEYXRhLnZhbHVlcyA9IG51bGw7XG4gICAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgICAgdGhpcy51cGxvYWRXYWl0TXMgKz0gdXRpbC5ub3coKSAtIHN0YXJ0O1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBuZXdUZXh0dXJlID0gdGhpcy5hY3F1aXJlVGV4dHVyZSh0ZXhTaGFwZSwgdXNhZ2UsIGR0eXBlLCBpc1BhY2tlZCk7XG4gICAgICB0ZXhEYXRhLnRleHR1cmUgPSBuZXdUZXh0dXJlO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkOiBEYXRhSWQsIGZsb2F0MzJWYWx1ZXM/OiBGbG9hdDMyQXJyYXkpOlxuICAgICAgVHlwZWRBcnJheSB7XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7ZHR5cGV9ID0gdGV4RGF0YTtcblxuICAgIHRoaXMucmVsZWFzZUdQVURhdGEoZGF0YUlkKTtcblxuICAgIGlmIChmbG9hdDMyVmFsdWVzICE9IG51bGwpIHtcbiAgICAgIHRleERhdGEudmFsdWVzID0gZmxvYXQzMlRvVHlwZWRBcnJheShmbG9hdDMyVmFsdWVzLCBkdHlwZSBhcyAnZmxvYXQzMicpO1xuICAgIH1cbiAgICByZXR1cm4gdGV4RGF0YS52YWx1ZXMgYXMgVHlwZWRBcnJheTtcbiAgfVxuXG4gIHByaXZhdGUgYWNxdWlyZVRleHR1cmUoXG4gICAgICB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSwgdGV4VHlwZTogVGV4dHVyZVVzYWdlLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgICBpc1BhY2tlZDogYm9vbGVhbik6IFdlYkdMVGV4dHVyZSB7XG4gICAgdGhpcy5udW1CeXRlc0luR1BVICs9IHRoaXMuY29tcHV0ZUJ5dGVzKHRleFNoYXBlLCBkdHlwZSk7XG4gICAgaWYgKCF0aGlzLndhcm5lZEFib3V0TWVtb3J5ICYmXG4gICAgICAgIHRoaXMubnVtQnl0ZXNJbkdQVSA+IHRoaXMubnVtTUJCZWZvcmVXYXJuaW5nICogMTAyNCAqIDEwMjQpIHtcbiAgICAgIGNvbnN0IG1iID0gKHRoaXMubnVtQnl0ZXNJbkdQVSAvIDEwMjQgLyAxMDI0KS50b0ZpeGVkKDIpO1xuICAgICAgdGhpcy53YXJuZWRBYm91dE1lbW9yeSA9IHRydWU7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYEhpZ2ggbWVtb3J5IHVzYWdlIGluIEdQVTogJHttYn0gTUIsIGAgK1xuICAgICAgICAgIGBtb3N0IGxpa2VseSBkdWUgdG8gYSBtZW1vcnkgbGVha2ApO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy50ZXh0dXJlTWFuYWdlci5hY3F1aXJlVGV4dHVyZSh0ZXhTaGFwZSwgdGV4VHlwZSwgaXNQYWNrZWQpO1xuICB9XG5cbiAgcHJpdmF0ZSBjb21wdXRlQnl0ZXMoc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIGR0eXBlOiBEYXRhVHlwZSkge1xuICAgIHJldHVybiBzaGFwZVswXSAqIHNoYXBlWzFdICogdXRpbC5ieXRlc1BlckVsZW1lbnQoZHR5cGUpO1xuICB9XG59XG5cbmZ1bmN0aW9uIGZsb2F0MzJUb1R5cGVkQXJyYXk8RCBleHRlbmRzIE51bWVyaWNEYXRhVHlwZT4oXG4gICAgYTogRmxvYXQzMkFycmF5LCBkdHlwZTogRCk6IHRmLkRhdGFUeXBlTWFwW0RdIHtcbiAgaWYgKGR0eXBlID09PSAnZmxvYXQzMicgfHwgZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIGEgYXMgdGYuRGF0YVR5cGVNYXBbRF07XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicgfHwgZHR5cGUgPT09ICdib29sJykge1xuICAgIGNvbnN0IHJlc3VsdCA9IChkdHlwZSA9PT0gJ2ludDMyJykgPyBuZXcgSW50MzJBcnJheShhLmxlbmd0aCkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXcgVWludDhBcnJheShhLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXN1bHQubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFtpXSA9IE1hdGgucm91bmQoYVtpXSk7XG4gICAgfVxuICAgIHJldHVybiByZXN1bHQgYXMgdGYuRGF0YVR5cGVNYXBbRF07XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIGR0eXBlICR7ZHR5cGV9YCk7XG4gIH1cbn1cbiJdfQ==