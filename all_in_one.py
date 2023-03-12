import argparse
import glob
import os
import pdb
import pickle
import subprocess
import time
from shutil import rmtree

import numpy as np
from cv2 import cv2
from scenedetect import VideoManager, StatsManager, SceneManager, ContentDetector
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from tqdm import tqdm

from SyncNetInstance import SyncNetInstance
from detectors import S3FD


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video():
    DET = S3FD(device='cuda')

    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, '*.jpg'))
    flist.sort()

    dets = []

    for fidx, fname in tqdm(enumerate(flist), total=len(flist)):
        # start_time = time.time()
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
        # elapsed_time = time.time() - start_time
        # print('%s-%05d; %d dets; %.2f Hz' % (
        #     os.path.join(opt.avi_dir, opt.reference, 'video.avi'), fidx, len(dets[-1]), (1 / elapsed_time)))
    # savepath = os.path.join(opt.work_dir, opt.reference, 'faces.pckl')
    # with open(savepath, 'wb') as fil:
    #     pickle.dump(dets, fil)

    return dets


# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========
def scene_detect():
    video_manager = VideoManager([os.path.join(opt.avi_dir, opt.reference, 'video.avi')])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()

    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode)

    savepath = os.path.join(opt.work_dir, opt.reference, 'scene.pckl')

    if not scene_list:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

    with open(savepath, 'wb') as fil:
        pickle.dump(scene_list, fil)

    print('%s - scenes detected %d' % (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), len(scene_list)))

    return scene_list


# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========
def track_shot(scenefaces):
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []

    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        elif len(track) > opt.min_track:
            framenum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frame_i = np.arange(framenum[0], framenum[-1] + 1)
            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    return tracks


# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
def crop_video(track, cropfile):
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, '*.jpg'))
    flist.sort()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile + 't.avi', fourcc, opt.frame_rate, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)  # crop center x
        dets['x'].append((det[0] + det[2]) / 2)  # crop center y

    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = opt.crop_scale
        bs = dets['s'][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audiotmp = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
    audiostart = (track['frame'][0]) / opt.frame_rate
    audioend = (track['frame'][-1] + 1) / opt.frame_rate
    vOut.release()

    # ========== CROP AUDIO FILE ==========
    command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (
        os.path.join(opt.avi_dir, opt.reference, 'audio.wav'), audiostart, audioend, audiotmp))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        pdb.set_trace()
    sample_rate, audio = wavfile.read(audiotmp)
    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile, audiotmp, cropfile))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        pdb.set_trace()
    print('Written %s' % cropfile)
    os.remove(cropfile + 't.avi')
    print('Mean pos: x %.2f y %.2f s %.2f' % (np.mean(dets['x']), np.mean(dets['y']), np.mean(dets['s'])))
    return {'track': track, 'proc_track': dets}


def run_pipline():
    # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (
        opt.videofile, os.path.join(opt.avi_dir, opt.reference, 'video.avi')))
    subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (
        os.path.join(opt.avi_dir, opt.reference, 'video.avi'), os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (
        os.path.join(opt.avi_dir, opt.reference, 'video.avi'), os.path.join(opt.avi_dir, opt.reference, 'audio.wav')))
    subprocess.call(command, shell=True, stdout=None)

    # ========== FACE DETECTION ==========
    faces = inference_video()

    # ========== SCENE DETECTION ==========
    scene = scene_detect()

    # ========== FACE TRACKING ==========
    alltracks = []
    vidtracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
            alltracks.extend(track_shot(faces[shot[0].frame_num:shot[1].frame_num]))

    # ========== FACE TRACK CROP ==========
    for ii, track in enumerate(alltracks):
        vidtracks.append(crop_video(track, os.path.join(opt.crop_dir, opt.reference, '%05d' % ii)))

    # ========== SAVE RESULTS ==========
    # savepath = os.path.join(opt.work_dir, opt.reference, 'tracks.pckl')
    # with open(savepath, 'wb') as fil:
    #     pickle.dump(vidtracks, fil)
    rmtree(os.path.join(opt.tmp_dir, opt.reference))
    return vidtracks


def run_syncnet():
    # ==================== LOAD MODEL AND FILE LIST ====================
    s = SyncNetInstance()
    s.loadParameters(opt.initial_model)
    print("Model %s loaded." % opt.initial_model)
    flist = glob.glob(os.path.join(opt.crop_dir, opt.reference, '0*.avi'))
    flist.sort()

    # ==================== GET OFFSETS ====================
    dists = []
    for idx, fname in enumerate(flist):
        offset, conf, dist = s.evaluate(opt, videofile=fname)
        dists.append(dist)

    # ==================== PRINT RESULTS TO FILE ====================
    # with open(os.path.join(opt.work_dir, opt.reference, 'activesd.pckl'), 'wb') as fil:
    # pickle.dump(dists, fil)

    return dists


def run_visualise(tracks, dists):
    # ==================== LOAD FILES ====================
    # with open(os.path.join(opt.work_dir, opt.reference, 'tracks.pckl'), 'rb') as fil:
    #     tracks = pickle.load(fil, encoding='latin1')
    # with open(os.path.join(opt.work_dir, opt.reference, 'activesd.pckl'), 'rb') as fil:
    #     dists = pickle.load(fil, encoding='latin1')
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, '*.jpg'))
    flist.sort()

    # ==================== SMOOTH FACES ====================

    faces = [[] for i in range(len(flist))]

    for tidx, track in enumerate(tracks):

        mean_dists = np.mean(np.stack(dists[tidx], 1), 1)
        minidx = np.argmin(mean_dists, 0)
        minval = mean_dists[minidx]

        fdist = np.stack([dist[minidx] for dist in dists[tidx]])
        fdist = np.pad(fdist, (3, 3), 'constant', constant_values=10)

        fconf = np.median(mean_dists) - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)

        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            faces[frame].append(
                {'track': tidx, 'conf': fconfm[fidx], 's': track['proc_track']['s'][fidx],
                 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})

        # ==================== ADD DETECTIONS TO VIDEO ====================
        first_image = cv2.imread(flist[0])
        fw = first_image.shape[1]
        fh = first_image.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vOut = cv2.VideoWriter(os.path.join(opt.avi_dir, opt.reference, 'video_only.avi'), fourcc, opt.frame_rate,
                               (fw, fh))
        for fidx, fname in tqdm(enumerate(flist), total=len(flist)):
            image = cv2.imread(fname)
            # for face in faces[fidx]:
            #     clr = max(min(face['conf'] * 25, 255), 0)
            #     cv2.rectangle(image, (int(face['x'] - face['s']), int(face['y'] - face['s'])),
            #                   (int(face['x'] + face['s']), int(face['y'] + face['s'])), (0, clr, 255 - clr), 3)
            #     cv2.putText(image, 'Track %d, Conf %.3f' % (face['track'], face['conf']),
            #                 (int(face['x']-face['s']), int(face['y']-face['s'])),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            vOut.write(image)
            # print('Frame %d' % fidx)
        vOut.release()

        # ========== COMBINE AUDIO AND VIDEO FILES ==========
        command = ("ffmpeg -y -i %s -i %s -c:v copy -c:a copy %s" % (
            os.path.join(opt.avi_dir, opt.reference, 'video_only.avi'),
            os.path.join(opt.avi_dir, opt.reference, 'audio.wav'),
            os.path.join(opt.avi_dir, opt.reference, 'video_out.avi')))  # -async 1
        subprocess.call(command, shell=True, stdout=None)


def main():
    setattr(opt, 'avi_dir', os.path.join(opt.data_dir, 'pyavi'))
    setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
    setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
    setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))
    setattr(opt, 'frames_dir', os.path.join(opt.data_dir, 'pyframes'))
    # ========== DELETE EXISTING DIRECTORIES ==========
    if os.path.exists(os.path.join(opt.work_dir, opt.reference)):
        rmtree(os.path.join(opt.work_dir, opt.reference))
    if os.path.exists(os.path.join(opt.crop_dir, opt.reference)):
        rmtree(os.path.join(opt.crop_dir, opt.reference))
    if os.path.exists(os.path.join(opt.avi_dir, opt.reference)):
        rmtree(os.path.join(opt.avi_dir, opt.reference))
    if os.path.exists(os.path.join(opt.frames_dir, opt.reference)):
        rmtree(os.path.join(opt.frames_dir, opt.reference))
    if os.path.exists(os.path.join(opt.tmp_dir, opt.reference)):
        rmtree(os.path.join(opt.tmp_dir, opt.reference))
    # ========== MAKE NEW DIRECTORIES ==========

    os.makedirs(os.path.join(opt.work_dir, opt.reference))
    os.makedirs(os.path.join(opt.crop_dir, opt.reference))
    os.makedirs(os.path.join(opt.avi_dir, opt.reference))
    os.makedirs(os.path.join(opt.frames_dir, opt.reference))
    os.makedirs(os.path.join(opt.tmp_dir, opt.reference))
    # RUN PIPLINE
    tracks = run_pipline()
    # RUN SYNCNET
    dists = run_syncnet()
    # RUN VISUALISE
    run_visualise(tracks, dists)
    pass


def run(videofile):
    setattr(opt, 'videofile', videofile)
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FaceTracker")
    parser.add_argument('--data_dir', type=str, default='data/work', help='Output direcotry')
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--videofile', type=str, default='', help='Input video file')
    parser.add_argument('--reference', type=str, default='', help='Video reference')
    parser.add_argument('--facedet_scale', type=float, default=0.25, help='Scale factor for face detection')
    parser.add_argument('--crop_scale', type=float, default=0.40, help='Scale bounding box')
    parser.add_argument('--min_track', type=int, default=20, help='Minimum facetrack duration')
    parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
    parser.add_argument('--num_failed_det', type=int, default=25,
                        help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--min_face_size', type=int, default=100, help='Minimum face size in pixels')
    opt = parser.parse_args()

    main()
