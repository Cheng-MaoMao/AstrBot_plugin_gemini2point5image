import json
import os
from datetime import datetime
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.message_components import Image, Plain
from astrbot.core.message.components import Reply
from .utils.ttp import generate_image_openrouter
from .utils.file_send_server import send_file

@register("gemini-25-image-openrouter", "喵喵", "使用openrouter的免费api生成图片", "2.2")
class MyPlugin(Star):
    """
    一个使用OpenRouter API和Gemini模型生成图像的AstrBot插件。
    支持多API密钥轮换、自定义模型、参考图生成、绘图次数限制等功能。
    """
    def __init__(self, context: Context, config: dict):
        """
        插件初始化函数。
        - 加载配置，包括API密钥、模型名称、服务器地址等。
        - 初始化绘图次数限制功能，加载历史使用记录。
        """
        super().__init__(context)
        # --- API与模型配置 ---
        self.openrouter_api_keys = config.get("openrouter_api_keys", [])
        old_api_key = config.get("openrouter_api_key")
        if old_api_key and not self.openrouter_api_keys:
            self.openrouter_api_keys = [old_api_key]
        
        self.custom_api_base = config.get("custom_api_base", "").strip()
        self.model_name = config.get("model_name", "google/gemini-2.5-flash-image-preview:free").strip()
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        
        # --- 文件传输服务器配置 ---
        self.nap_server_address = config.get("nap_server_address")
        self.nap_server_port = config.get("nap_server_port")

        # --- 绘图次数限制功能配置 ---
        self.rate_limit_config = config.get("rate_limit", {})
        self.usage_records = {}
        self.usage_records_path = os.path.join("data", "gemini-25-image-openrouter", "usage_records.json")
        self._load_usage_records()
        
        self._global_config_loaded = False

    async def _load_global_config(self):
        """异步加载全局配置"""
        if self._global_config_loaded:
            return
        try:
            plugin_config = await sp.global_get("gemini-25-image-openrouter", {})
            if "custom_api_base" in plugin_config:
                self.custom_api_base = plugin_config["custom_api_base"]
                logger.info(f"从全局配置加载 custom_api_base: {self.custom_api_base}")
            if "model_name" in plugin_config:
                self.model_name = plugin_config["model_name"]
                logger.info(f"从全局配置加载 model_name: {self.model_name}")
            self._global_config_loaded = True
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
            self._global_config_loaded = True

    def _load_usage_records(self):
        """
        从JSON文件加载绘图使用记录到内存。
        - 如果记录文件存在，则读取内容。
        - 如果文件或目录不存在，则会自动创建，确保首次启动时能正常工作。
        - 捕获并记录任何可能的异常，防止因文件问题导致插件加载失败。
        """
        try:
            if os.path.exists(self.usage_records_path):
                with open(self.usage_records_path, "r", encoding="utf-8") as f:
                    self.usage_records = json.load(f)
                    logger.info("成功加载绘图使用记录。")
            else:
                os.makedirs(os.path.dirname(self.usage_records_path), exist_ok=True)
                logger.info(f"使用记录文件不存在，已创建目录: {os.path.dirname(self.usage_records_path)}")
        except Exception as e:
            logger.error(f"加载绘图使用记录失败: {e}")

    def _save_usage_records(self):
        """
        将内存中的使用记录保存到JSON文件。将内存中的使用记录保存到JSON文件。
        - 使用json.dump进行序列化，并设置格式使其更具可读性。
        - 捕获并记录任何可能的异常，如磁盘无空间或无写入权限。
        """
        try:
            with open(self.usage_records_path, "w", encoding="utf-8") as f:
                json.dump(self.usage_records, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存绘图使用记录失败: {e}")

    async def _check_and_update_limit(self, event: AstrMessageEvent) -> tuple[bool, str]:
        """
        检查并更新用户的绘图次数限制。采用“黑名单”模式。
        """
        if not self.rate_limit_config.get("enabled", False):
            return True, ""

        user_id = event.get_sender_id()
        group_id = event.get_group_id()
        
        limit = -1
        limit_type = None
        
        user_limits = self.rate_limit_config.get("user_limits", [])
        if isinstance(user_limits, list):
            user_limit_config = next((item for item in user_limits if isinstance(item, dict) and item.get("user_id") == user_id), None)
            if user_limit_config:
                limit = user_limit_config.get("limit", self.rate_limit_config.get("default_limit", 10))
                limit_type = "用户"
        else:
            logger.warning("配置中的 user_limits 格式不正确，应为列表")

        if limit_type is None and group_id:
            group_limits = self.rate_limit_config.get("group_limits", [])
            if isinstance(group_limits, list):
                group_limit_config = next((item for item in group_limits if isinstance(item, dict) and item.get("group_id") == group_id), None)
                if group_limit_config:
                    limit = group_limit_config.get("limit", self.rate_limit_config.get("default_limit", 10))
                    limit_type = "群组"
            else:
                logger.warning("配置中的 group_limits 格式不正确，应为列表")

        if limit_type is None:
            return True, ""

        reset_interval = self.rate_limit_config.get("reset_interval_minutes", 1440)
        current_timestamp = int(datetime.now().timestamp())
        record_key = f"group_{group_id}" if group_id else f"user_{user_id}"

        user_record = self.usage_records.get(record_key, {"timestamp": current_timestamp, "count": 0})

        time_since_last_draw = (current_timestamp - user_record.get("timestamp", 0)) / 60
        if time_since_last_draw > reset_interval:
            user_record = {"timestamp": current_timestamp, "count": 0}

        if user_record["count"] >= limit:
            return False, f"您已达到绘图次数上限（{limit}次），请稍后再试。"

        user_record["count"] += 1
        user_record["timestamp"] = current_timestamp
        self.usage_records[record_key] = user_record
        self._save_usage_records()

        remaining = limit - user_record["count"]
        
        if limit_type == "用户":
            message = f"用户({user_id})绘图成功！您还剩余 {remaining} 次绘图机会。"
        else:
            message = f"绘图成功！您还剩余 {remaining} 次绘图机会。"
            
        return True, message

    async def send_image_with_callback_api(self, image_path: str) -> Image:
        """
         优先使用callback_api_base发送图片，失败则退回到本地文件发送
        
        Args:
            image_path (str): 图片文件路径
            
        Returns:
            Image: 图片组件
        """
        callback_api_base = self.context.get_config().get("callback_api_base")
        if not callback_api_base:
            logger.info("未配置callback_api_base，使用本地文件发送")
            return Image.fromFileSystem(image_path)

        logger.info(f"检测到配置了callback_api_base: {callback_api_base}")
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.convert_to_web_link()
            logger.info(f"成功生成下载链接: {download_url}")
            return Image.fromURL(download_url)
        except (IOError, OSError) as e:
            logger.warning(f"文件操作失败: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"网络连接失败: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)
        except Exception as e:
            logger.error(f"发送图片时出现未预期的错误: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)

    @filter.llm_tool(name="gemini-pic-gen")
    async def pic_gen(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True):
        """
             Generate or modify images using the Gemini model via the OpenRouter API.
            When a user requests image generation or drawing, call this function.
            If use_reference_images is True and the user has provided images in their message,
            those images will be used as references for generation or modification.
            If no images are provided or use_reference_images is False, pure text-to-image generation will be performed.

            Here are some examples:
            1. If the user wants to generate a large figure model, such as an anime character with normal proportions, please use a prompt like:
            "Please accurately transform the main subject in this photo into a realistic, masterpiece-like 1/7 scale PVC statue.
            A box should be placed beside the statue: the front of the box should have a large, clear transparent window printed with the main artwork, product name, brand logo, barcode, and a small specification or authenticity verification panel. A small price tag sticker must also be attached to the corner of the box. Meanwhile, a computer monitor should be placed at the back, and the monitor screen needs to display the ZBrush modeling process of this statue.
            In front of the packaging box, the statue should be placed on a round plastic base. The statue must have 3D dimensionality and a sense of realism, and the texture of the PVC material needs to be clearly represented. If the background can be set as an indoor scene, the effect will be even better.

            2. If the user wants to generate a chibi figure model or a small, cute figure, please use a prompt like:
            "Please accurately transform the main subject in this photo into a realistic, masterpiece-like 1/7 scale PVC statue.
            Behind the side of this statue, a box should be placed: on the front of the box, the original image I entered, with the themed artwork, product name, brand logo, barcode, and a small specification or authenticity verification panel. A small price tag sticker must also be attached to one corner of the box. Meanwhile, a computer monitor should be placed at the back, and the monitor screen needs to display the ZBrush modeling process of this statue.
            In front of the packaging box, the statue should be placed on a round plastic base. The statue must have 3D dimensionality and a sense of realism, and the texture of the PVC material needs to be clearly represented. If the background can be set as an indoor scene, the effect will be even better.

            Below are detailed guidelines to note:
            When repairing any missing parts, there must be no poorly executed elements.
            When repairing human figures (if applicable), the body parts must be natural, movements must be coordinated, and the proportions of all parts must be reasonable.
            If the original photo is not a full-body shot, try to supplement the statue to make it a full-body version.
            The human figure's expression and movements must be exactly consistent with those in the photo.
            The figure's head should not appear too large, its legs should not appear too short, and the figure should not look stunted—this guideline may be ignored if the statue is a chibi-style design.
            For animal statues, the realism and level of detail of the fur should be reduced to make it more like a statue rather than the real original creature.
            No outer outline lines should be present, and the statue must not be flat.
            Please pay attention to the perspective relationship of near objects appearing larger and far objects smaller."

            Args:
            - image_description (string): Description of the image to generate. Translate to English can be better.
            - use_reference_images (bool): Whether to use images from the user's message as reference. Default True.
        """
        can_draw, message = await self._check_and_update_limit(event)
        if not can_draw:
            yield event.plain_result(message)
            return

        await self._load_global_config()
        
        input_images = []
        if use_reference_images:
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
                for comp in event.message_obj.message:
                    if isinstance(comp, Image):
                        try:
                            input_images.append(await comp.convert_to_base64())
                        except Exception as e:
                            logger.warning(f"转换当前消息中的参考图片到base64失败: {e}")
                    elif isinstance(comp, Reply) and comp.chain:
                        for reply_comp in comp.chain:
                            if isinstance(reply_comp, Image):
                                try:
                                    input_images.append(await reply_comp.convert_to_base64())
                                except Exception as e:
                                    logger.warning(f"转换引用消息中的参考图片到base64失败: {e}")
            
            if input_images:
                logger.info(f"使用了 {len(input_images)} 张参考图片进行图像生成")
            else:
                logger.info("未找到参考图片，执行纯文本图像生成")

        try:
            image_url, image_path = await generate_image_openrouter(
                image_description, self.openrouter_api_keys, model=self.model_name,
                input_images=input_images, api_base=self.custom_api_base or None,
                max_retry_attempts=self.max_retry_attempts
            )
            
            if not image_url or not image_path:
                yield event.plain_result("图像生成失败，请检查API配置和网络连接。")
                return
            
            if self.nap_server_address and self.nap_server_address != "localhost":
                image_path = await send_file(image_path, host=self.nap_server_address, port=self.nap_server_port)
            
            image_component = await self.send_image_with_callback_api(image_path)
            chain = [image_component, Plain(message)] if message else [image_component]
            yield event.chain_result(chain)
            
        except Exception as e:
            logger.error(f"图像生成过程出现未预期的错误: {e}")
            yield event.plain_result(f"图像生成失败: {str(e)}")

    @filter.command_group("banana")
    def banan(self):
        """OpenRouter绘图插件快速配置命令组"""
        pass

    @banan.command("baseurl")
    async def switch_base_url(self, event: AstrMessageEvent, new_base_url: str = None, save_global: str = "false"):
        """快速切换openrouter绘图插件的base URL
        
        使用方法:
        /banan baseurl - 查看当前base URL
        /banan baseurl <新的base_url> - 临时切换base URL（会话级别）
        /banan baseurl <新的base_url> true - 永久切换base URL（全局配置）
        """
        # 确保加载最新的全局配置
        await self._load_global_config()
        
        if not new_base_url:
            current_url = self.custom_api_base or "https://openrouter.ai/api/v1"
            yield event.plain_result(f"当前 base URL: {current_url}\n使用方法:\n/banan baseurl <新的base_url> [true] - true表示永久保存")
            return
        
        self.custom_api_base = new_base_url.strip()
        
        if save_global.lower() in ["true", "1", "yes", "y"]:
            try:
                plugin_config = await sp.global_get("gemini-25-image-openrouter", {})
                plugin_config["custom_api_base"] = self.custom_api_base
                await sp.global_put("gemini-25-image-openrouter", plugin_config)
                yield event.plain_result(f"已永久切换 base URL 到: {self.custom_api_base}")
            except Exception as e:
                logger.error(f"保存全局配置失败: {e}")
                yield event.plain_result(f"已临时切换 base URL，但保存全局配置失败: {str(e)}")
        else:
            yield event.plain_result(f"已临时切换 base URL 到: {self.custom_api_base}")

    @banan.command("model")
    async def switch_model(self, event: AstrMessageEvent, new_model: str = None, save_global: str = "false"):
        """快速切换openrouter绘图插件的模型
        
        使用方法:
        /banan model - 查看当前模型
        /banan model <模型名> - 临时切换模型（会话级别）
        /banan model <模型名> true - 永久切换模型（全局配置）
        
        例如: /banan model google/gemini-2.5-flash-image-preview:free
        """
        # 确保加载最新的全局配置
        await self._load_global_config()
        
        if not new_model:
            yield event.plain_result(f"当前模型: {self.model_name}\n使用方法:\n/banan model <模型名> [true] - true表示永久保存")
            return
        
        self.model_name = new_model.strip()
        
        if save_global.lower() in ["true", "1", "yes", "y"]:
            try:
                plugin_config = await sp.global_get("gemini-25-image-openrouter", {})
                plugin_config["model_name"] = self.model_name
                await sp.global_put("gemini-25-image-openrouter", plugin_config)
                yield event.plain_result(f"已永久切换模型到: {self.model_name}")
            except Exception as e:
                logger.error(f"保存全局配置失败: {e}")
                yield event.plain_result(f"已临时切换模型，但保存全局配置失败: {str(e)}")
        else:
            yield event.plain_result(f"已临时切换模型到: {self.model_name}")

    @filter.command("手办化")
    async def figure_transform(self, event: AstrMessageEvent):
        """将用户提供的图片转换为手办效果"""
        can_draw, message = await self._check_and_update_limit(event)
        if not can_draw:
            yield event.plain_result(message)
            return
            
        await self._load_global_config()
        
        input_images = []
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    try:
                        input_images.append(await comp.convert_to_base64())
                    except Exception as e:
                        logger.warning(f"转换图片到base64失败: {e}")
                elif isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            try:
                                input_images.append(await reply_comp.convert_to_base64())
                            except Exception as e:
                                logger.warning(f"转换引用消息中的图片到base64失败: {e}")
        
        if not input_images:
            yield event.plain_result("请提供一张图片以进行手办化处理！")
            return
        
        logger.info(f"开始手办化处理，使用了 {len(input_images)} 张图片")
        
        figure_prompt =  """Please accurately transform the main subject in this image into a realistic, masterpiece-quality 1/7 scale PVC figure.

Specific Requirements:
1. **Figure Creation**: Convert the subject into a high-quality PVC figure with obvious three-dimensional depth and the characteristic glossy finish of PVC material
2. **Packaging Box Design**: Place an exquisite packaging box beside the figure. The front of the box should have a large transparent window displaying the original image, along with brand logos, product name, barcode, and detailed specification panels
3. **Display Base**: The figure should be placed on a round, transparent plastic base with visible thickness
4. **Background Setup**: Place a computer monitor in the background, with the screen displaying the ZBrush 3D modeling process of this figure
5. **Indoor Scene**: Set the entire scene in an indoor environment with appropriate lighting effects

Technical Requirements:
- Maintain the exact characteristics, expressions, and poses from the original image
- The figure must have obvious three-dimensional effects and must never appear flat
- PVC material texture should be clearly visible and realistic
- Avoid any cartoon outline strokes
- If the original image is not full-body, complete it as a full-body figure
- Character proportions should be natural and coordinated (head not too large, legs not too short)
- For animal figures, reduce fur realism to make it more statue-like rather than the real creature
- Pay attention to perspective relationships with near objects appearing larger and distant objects smaller
- No outer outline lines should be present

Please ensure the final result looks like a real commercial figure product that could exist in the market."""

        try:
            image_url, image_path = await generate_image_openrouter(
                figure_prompt, self.openrouter_api_keys, model=self.model_name,
                input_images=input_images, api_base=self.custom_api_base or None,
                max_retry_attempts=self.max_retry_attempts
            )
            
            if not image_url or not image_path:
                yield event.plain_result("手办化处理失败，请检查API配置和网络连接。")
                return
            
            if self.nap_server_address and self.nap_server_address != "localhost":
                image_path = await send_file(image_path, host=self.nap_server_address, port=self.nap_server_port)
            
            image_component = await self.send_image_with_callback_api(image_path)
            result_chain = [Plain("✨ 手办化处理完成！"), image_component, Plain(message)] if message else [Plain("✨ 手办化处理完成！"), image_component]
            yield event.chain_result(result_chain)
            
        except Exception as e:
            logger.error(f"手办化处理过程出现未预期的错误: {e}")
            yield event.plain_result(f"手办化处理失败: {str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("reset_images")
    async def reset_images_command(self, event: AstrMessageEvent):
        """重置所有用户和群组的绘画次数"""
        self.usage_records = {}
        self._save_usage_records()
        yield event.plain_result("所有用户和群组的绘图次数已成功重置。")
