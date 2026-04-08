def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    
    CAM_POINTS = np.array([
            [ 0,   0,   0],
            [-1,  -1, 1.5],
            [ 1,  -1, 1.5],
            [ 1,   1, 1.5],
            [-1,   1, 1.5],
            [-0.5, 1, 1.5],
            [ 0.5, 1, 1.5],
            [ 0, 1.2, 1.5]]) / 3

    CAM_LINES = np.array([
        [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])


    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def vis_gs(pc, cam=None, save=False, mask=None):
    from utils.sh_utils import eval_sh
    pcd = o3d.geometry.PointCloud()
    
    if mask is not None:
        shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)[mask]
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz[mask].detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    else:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz.detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    if cam is not None:
        total_list = [pcd]
        if isinstance(cam, list):
            for c in cam:
                cam_actor = create_camera_actor(True)
                ref_pose = np.eye(4)
                R, T = c.R, c.T
                ref_pose[:3, :3] = R
                ref_pose[:3, 3] = T
                
                ref_pose = np.linalg.inv(ref_pose)
                cam_actor.transform(ref_pose)
                # cam_actor.rotate(R)
                # cam_actor.translate(T)
                total_list.append(cam_actor)
        else :
            cam_actor = create_camera_actor(True)
            c = cam
            ref_pose = np.eye(4)
            R, T = c.R, c.T
            ref_pose[:3, :3] = R
            ref_pose[:3, 3] = T
            
            ref_pose = np.linalg.inv(ref_pose)
            cam_actor.transform(ref_pose)
            # cam_actor.rotate(R)
            # cam_actor.translate(T)
            total_list.append(cam_actor)
        return o3d.visualization.draw_geometries(total_list)
    
    if save:
        o3d.io.write_point_cloud("test.ply", pcd)
        
    o3d.visualization.draw_geometries([pcd])