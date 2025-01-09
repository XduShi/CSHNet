def forward(self,x):
        ####不采用cat
        x = self.model1(x) #1*64*256*256
        #visual init64
        init = x.squeeze(0)
        init_1 = 0
        for i in range(64):
            init_1 = init.data[i]
            util.save_image(util.tensor2im(init_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/initial64/initial64_'+str(i)+'.png')
        
        #x = self.model_D(x) 
        ##visual d1
        x_d1 = self.model_D[0](x)
        #x_d11 = self.model_D[0].down(x)
        #print(x_d11.shape)
        #x_d11 = self.model_D[0].norm_layer(x_d11)
        #x_d11_r = self.model_D[0].activation(x_d11)
        d1 = x_d1.squeeze(0)
        d1_1 = 0
        for i in range(128):
            d1_1 = d1.data[i]
            util.save_image(util.tensor2im(d1_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/d1_pool/d1_pool_'+str(i)+'.png')
        ##visual d2
       
        x_d2 = self.model_D[1](x_d1)
        #x_d22 = self.model_D[1].down(x_d11_r)
        #x_d22 = self.model_D[1].norm_layer(x_d22)
        #x_d22_r = self.model_D[1].activation(x_d22)
        d2 = x_d2.squeeze(0)
        d2_1 = 0
        for i in range(256):
            d2_1 = d2.data[i]
            util.save_image(util.tensor2im(d2_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/d2_pool/d2_pool_'+str(i)+'.png')
        #x = self.model_M(x) #1*256*64*64
        ##visual m1
        
        x_m1 = self.model_M[0](x_d2)
        m1 = x_m1.squeeze(0)
        m1_1 = 0
        for i in range(256):
            m1_1 = m1.data[i]
            util.save_image(util.tensor2im(m1_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/m1/m1_'+str(i)+'.png')
        
        ##visual m2
        x_m2 = self.model_M[1](x_m1)
        m2 = x_m2.squeeze(0)
        m2_1 = 0
        for i in range(256):
            m2_1 = m2.data[i]
            util.save_image(util.tensor2im(m2_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/m2/m2_'+str(i)+'.png')
    
        ##visual m3
        x_m3 = self.model_M[2](x_m2)
        m3 = x_m3.squeeze(0)
        m3_1 = 0
        for i in range(256):
            m3_1 = m3.data[i]
            util.save_image(util.tensor2im(m3_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/m3/m3_'+str(i)+'.png')
        
        ##visual m4
        x_m4 = self.model_M[3](x_m3)
        m4 = x_m4.squeeze(0)
        m4_1 = 0
        for i in range(256):
            m4_1 = m4.data[i]
            util.save_image(util.tensor2im(m4_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/m4/m4_'+str(i)+'.png')
        
        #x = self.model_U(x)
        ##visual u1
        x_u1 = self.model_U[0](x_m4)
        u1 = x_u1.squeeze(0)
        u1_1 = 0
        for i in range(128):
            u1_1 = u1.data[i]
            util.save_image(util.tensor2im(u1_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/u1/u1_'+str(i)+'.png') 
        ##visual u2
        x_u2 = self.model_U[1](x_u1)
        u2 = x_u2.squeeze(0)
        u2_1 = 0
        for i in range(64):
            u2_1 = u2.data[i]
            util.save_image(util.tensor2im(u2_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net/u2/u2_'+str(i)+'.png')
        #x = self.model2(x)  #1*3*256*256
        out = self.model2(x_u2)
        return out
    
    
    
    
    
    ###可视化特征图的forward 不加relu ---woPF(不添加PoolFormer) --LG_net_ab_woPF_V
    def forward(self,x):
        a,b = self.low_feature(x)
        x1 = self.model1(x)
        x1_p = self.model1[0](x) ##padding
        x1_c = self.model1[1](x1_p) ##conv
        x1_n = self.model1[2](x1_c) ##normalization
        ##可视化x1
        init = x1_n.squeeze(0)
        init_1 = 0
        init_s = 0
        for i in range(64):
            init_1 = init.data[i]
            init_s += init.data[i]
            util.save_image(util.tensor2im(init_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab_woPF/init64/init64_woPF_'+str(i)+'.png')
        util.save_image(util.tensor2im(init_s.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab_woPF/init64/init64_woPF_sum.png')
        ###
        #x_d = self.model_D(x1)
        ##可视化d1(1*128*128*128)
        x_d1 = self.model_D[0](x1)
        x_d1_c = self.model_D[0].down(x1) ##conv
        x_d1_n = self.model_D[0].norm_layer(x_d1_c)
        d1 = x_d1_n.squeeze(0)
        d1_1 = 0
        d1_s = 0
        for i in range(128):
            d1_1 = d1.data[i]
            d1_s += d1.data[i]
            util.save_image(util.tensor2im(d1_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab_woPF/d1/d1_woPF_'+str(i)+'.png')
        util.save_image(util.tensor2im(d1_s.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab_woPF/d1/d1_woPF_sum.png')
        ###
        ##可视化d2(1*256*64*64)
        x_d2 = self.model_D[1](x_d1)
        x_d2_c = self.model_D[1].down(x_d1) ##conv
        x_d2_n = self.model_D[1].norm_layer(x_d2_c) ##normalization
        d2 = x_d2_n.squeeze(0)
        d2_1 = 0
        d2_s = 0
        for i in range(256):
            d2_1 = d2.data[i]
            d2_s += d2.data[i]
            util.save_image(util.tensor2im(d2_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab_woPF/d2/d2_woPF_'+str(i)+'.png')
        util.save_image(util.tensor2im(d2_s.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab_woPF/d2/d2_woPF_sum.png')
        ###
        #x_m = self.low_mid(x_d,a,b)
        x_m = self.low_mid(x_d2,a,b)
        x_u = self.model_U(x_m)
        x2 = self.model2(x_u)
        return x2
    
    ###可视化特征图的forward (添加PoolFormer) --LG_net_ab_V
    # def forward(self,x):
    #     a,b = self.low_feature(x)
    #     x1 = self.model1(x)
    #     x1_p = self.model1[0](x) ##padding
    #     x1_c = self.model1[1](x1_p) ##conv
    #     x1_n = self.model1[2](x1_c) ##normalization
    #     x1_p = self.model1[4](x1_n) ##poolformer
    #     ##可视化x1
    #     init = x1_p.squeeze(0)
    #     init_1 = 0
    #     init_s = 0
    #     for i in range(64):
    #         init_1 = init.data[i]
    #         init_s += init.data[i]
    #         util.save_image(util.tensor2im(init_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab/init64/init64_wPF_'+str(i)+'.png')
    #     util.save_image(util.tensor2im(init_s.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab/init64/init64_wPF_sum.png')
    #     ###
    #     #x_d = self.model_D(x1)
    #     ##可视化d1(1*128*128*128)
    #     x_d1 = self.model_D[0](x1)
    #     x_d1_c = self.model_D[0].down(x1) ##conv
    #     x_d1_n = self.model_D[0].norm_layer(x_d1_c)  ##normalization
    #     x_d1_p = self.model_D[0].pool(x_d1_n)  ##poolformer
    #     d1 = x_d1_p.squeeze(0)
    #     d1_1 = 0
    #     d1_s = 0
    #     for i in range(128):
    #         d1_1 = d1.data[i]
    #         d1_s += d1.data[i]
    #         util.save_image(util.tensor2im(d1_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab/d1/d1_wPF_'+str(i)+'.png')
    #     util.save_image(util.tensor2im(d1_s.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab/d1/d1_wPF_sum.png')
    #     ###
    #     ##可视化d2(1*256*64*64)
    #     x_d2 = self.model_D[1](x_d1)
    #     x_d2_c = self.model_D[1].down(x_d1) ##conv
    #     x_d2_n = self.model_D[1].norm_layer(x_d2_c) ##normalization
    #     x_d2_p = self.model_D[1].pool(x_d2_n) ##poolformer
    #     d2 = x_d2_p.squeeze(0)
    #     d2_1 = 0
    #     d2_s = 0
    #     for i in range(256):
    #         d2_1 = d2.data[i]
    #         d2_s += d2.data[i]
    #         util.save_image(util.tensor2im(d2_1.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab/d2/d2_wPF_'+str(i)+'.png')
    #     util.save_image(util.tensor2im(d2_s.unsqueeze(0)), '/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/visual_G/LG_net_ab/d2/d2_wPF_sum.png')
    #     ###
    #     #x_m = self.low_mid(x_d,a,b)
    #     x_m = self.low_mid(x_d2,a,b)
    #     x_u = self.model_U(x_m)
    #     x2 = self.model2(x_u)
    #     return x2