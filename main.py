import base64
import os
import uuid
import re  # 修改：新增：导入re模块用于正则表达式匹配

import aiohttp
import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("AstrBot_Plugins_Canvas", "长安某", "gemini画图工具", "1.2.0")
class GeminiImageGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        """Gemini 图片生成与编辑插件初始化"""
        super().__init__(context)
        self.config = config

        logger.info(f"插件配置加载成功: {self.config}")

        # 读取多密钥配置
        self.api_keys = self.config.get("gemini_api_keys", [])
        self.current_key_index = 0

        # 初始化图片保存目录
        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"已创建图片临时目录: {self.save_dir}")

        # 初始化配置
        self.api_base_url = self.config.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )

        if not self.api_keys:
            logger.error("未配置任何 Gemini API 密钥，请在插件配置中填写")

    def _get_current_api_key(self):
        """获取当前使用的 API 密钥"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_next_api_key(self):
        """切换到下一个 API 密钥"""
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"已切换到下一个 API 密钥（索引：{self.current_key_index}）")

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片"""
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result(
                "请输入图片描述（示例：/gemini_image 一只戴帽子的猫在月球上）"
            )
            return

        save_path = None

        try:
            # 修改：删除：移除“正在生成图片，请稍等...”的提示
            # yield event.plain_result("正在生成图片，请稍等...")
            image_data = await self._generate_image_with_retry(prompt)

            if not image_data:
                logger.error("生成失败：所有API密钥均尝试完毕")
                yield event.plain_result("生成失败：所有API密钥均尝试失败")
                return

            # 保存图片
            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"图片已保存至: {save_path}")

            # 发送图片
            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功，提示词: {prompt}")

        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            yield event.plain_result(f"生成失败: {str(e)}")

        finally:
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除临时图片: {save_path}")
                except Exception as e:
                    logger.warning(f"删除临时图片失败: {str(e)}")

    @filter.command("gemini_edit", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str):
        """仅支持：引用图片后发送指令编辑图片"""
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        # 图片提取逻辑
        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未找到图片，请先长按图片发送回复后重试")
            return

        # 图片编辑处理
        async for result in self._process_image_edit(event, prompt, image_path):
            yield result

    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(self, event: AstrMessageEvent, prompt: str):
        """编辑现有图片。当你需要编辑图片时，请使用此工具。

        Args:
            prompt(string): 编辑描述（例如：把猫咪改成黑色）
        """
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result("请提供编辑描述（例如：把猫咪改成黑色）")
            return

        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result(
                "未找到图片，请先长按图片并点击“回复”，再输入编辑指令"
            )
            return

        async for result in self._process_image_edit(event, prompt, image_path):
            yield result

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片，当你需要生成图片时请使用此工具。

        Args:
            prompt(string): 图片描述文本（例如：画只猫）
        """
        async for result in self.generate_image(event, prompt):
            yield result

    # 提取回复中图片
    async def _extract_image_from_reply(self, event: AstrMessageEvent):
        """从回复消息中提取图片并返回本地路径"""
        try:
            message_components = event.message_obj.message
            reply_component = None
            for comp in message_components:
                if isinstance(comp, Comp.Reply):
                    reply_component = comp
                    logger.info(f"检测到回复消息（ID：{comp.id}），提取被引用图片")
                    break

            if not reply_component:
                logger.warning("未检测到回复组件（用户未长按图片回复）")
                return None

            # 从回复的chain中提取Image组件
            image_component = None
            for quoted_comp in reply_component.chain:
                if isinstance(quoted_comp, Comp.Image):
                    image_component = quoted_comp
                    logger.info(
                        f"从回复中提取到图片组件（file：{image_component.file}）"
                    )
                    break

            if not image_component:
                logger.warning("回复中未包含图片组件")
                return None

            # 获取本地图片路径（自动处理下载/转换）
            image_path = await image_component.convert_to_file_path()
            logger.info(f"图片已处理为本地路径：{image_path}")
            return image_path

        except Exception as e:
            logger.error(f"提取图片失败: {str(e)}", exc_info=True)
            return None

    # 统一的图片编辑处理逻辑
    async def _process_image_edit(
        self, event: AstrMessageEvent, prompt: str, image_path: str
    ):
        """处理图片编辑的核心逻辑"""
        save_path = None
        try:
            # 修改：删除：移除“正在编辑图片，请稍等...”的提示
            # yield event.plain_result("正在编辑图片，请稍等...")

            # 调用带重试的编辑方法
            image_data = await self._edit_image_with_retry(prompt, image_path)

            if not image_data:
                yield event.plain_result("编辑失败：所有API密钥均尝试失败")
                return

            # 保存并发送编辑后的图片
            save_path = os.path.join(self.save_dir, f"{uuid.uuid4()}_edited.png")
            with open(save_path, "wb") as f:
                f.write(image_data)

            yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            logger.info(f"图片编辑完成并发送，提示词: {prompt}")

        except Exception as e:
            logger.error(f"图片编辑出错：{str(e)}")
            yield event.plain_result(f"图片编辑失败：{str(e)}")

        finally:
            # 清理临时文件
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info(f"已删除原始图片临时文件：{image_path}")
                except Exception as e:
                    logger.warning(f"删除原始图片失败：{str(e)}")

            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除编辑图临时文件：{save_path}")
                except Exception as e:
                    logger.warning(f"删除编辑图失败: {str(e)}")

    async def _edit_image_with_retry(self, prompt, image_path):
        """带重试逻辑的图片编辑方法"""
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试编辑图片（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await self._edit_image_manually(prompt, image_path, current_key)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有API密钥均尝试失败")

        return None

    async def _generate_image_with_retry(self, prompt):
        """带重试逻辑的图片生成方法"""
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试生成图片（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await self._generate_image_manually(prompt, current_key)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有API密钥均尝试失败")

        return None

    # 修改：新增：辅助函数，用于从不同文本格式中提取图片URL
    def _extract_image_url_from_text(self, text_content: str) -> str | None:
        """
        从文本内容中提取图片URL，支持Markdown、HTML、BBCode和直接URL格式。
        优先级：Markdown -> HTML -> BBCode -> 直接URL
        """
        
        # 1. Markdown: ![alt text](url)
        # 修改：修改逻辑：Markdown图片链接正则表达式，更精确匹配URL
        markdown_match = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', text_content)
        if markdown_match:
            url = markdown_match.group(1)
            if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                logger.debug(f"Extracted Markdown URL: {url}")
                return url

        # 2. HTML <img> tag: <img src="url">
        # 修改：新增：HTML图片链接正则表达式
        html_match = re.search(r'<img[^>]*src=["\'](https?://[^"\'\s]+?)["\'][^>]*>', text_content, re.IGNORECASE)
        if html_match:
            url = html_match.group(1)
            if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                logger.debug(f"Extracted HTML URL: {url}")
                return url

        # 3. BBCode [img] tag: [img]url[/img]
        # 修改：新增：BBCode图片链接正则表达式
        bbcode_match = re.search(r'\[img\](https?://[^\[\]\s]+?)\[/img\]', text_content, re.IGNORECASE)
        if bbcode_match:
            url = bbcode_match.group(1)
            if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                logger.debug(f"Extracted BBCode URL: {url}")
                return url

        # 4. Direct URL ending with image extension
        # 修改：修改逻辑：直接URL正则表达式，确保是图片文件
        direct_url_match = re.search(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp))', text_content, re.IGNORECASE)
        if direct_url_match:
            url = direct_url_match.group(1)
            logger.debug(f"Extracted direct URL: {url}")
            return url

        return None

    # 修改：新增：从URL下载图片的方法
    async def _download_image_from_url(self, url: str, save_path: str):
        """
        从给定的URL下载图片并保存到指定路径。
        """
        logger.info(f"尝试从URL下载图片: {url} 到 {save_path}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()  # 检查HTTP状态码，如果不是2xx则抛出异常
                    with open(save_path, "wb") as f:
                        # 异步写入文件，分块读取以处理大文件
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    logger.info(f"图片已成功从URL下载并保存到: {save_path}")
            except aiohttp.ClientError as e:
                logger.error(f"下载图片失败 (aiohttp error): {e}")
                raise
            except Exception as e:
                logger.error(f"下载图片时发生未知错误: {e}")
                raise

    async def _edit_image_manually(self, prompt, image_path, api_key):
        """使用异步请求编辑图片"""
        model_name = "gemini-2.0-flash-preview-image-generation"

        # 修正API地址格式
        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.info(f"异步请求地址：{endpoint}")

        headers = {"Content-Type": "application/json"}

        # 读取图片并转换为Base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = (
                base64.b64encode(image_bytes)
                .decode("utf-8")
                .replace("\n", "")
                .replace("\r", "")
            )

        # 构建请求参数
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": image_base64}}
                    ],
                },
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024,
            },
        }

        # 异步发送请求
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=endpoint, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()  # 异步读取响应文本
                        logger.error(
                            f"API编辑请求失败: HTTP {response.status}, 响应: {response_text}"
                        )
                        response.raise_for_status()

                    # 异步解析JSON响应
                    data = await response.json()
                    # 修改：新增：记录完整的API响应数据，以便调试
                    logger.debug(f"Gemini API edit response data: {data}")

            except Exception as e:
                logger.error(f"异步编辑请求失败: {str(e)}")
                raise  # 传递异常触发重试

        # 解析图片数据或URL
        image_data = None
        image_url = None

        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
                    image_data = base64.b64decode(base64_str)
                    logger.info("Successfully extracted image data from inlineData (edit).")
                    break
                # 修改：修改逻辑：调用辅助函数识别多种格式的图片URL
                elif "text" in part:
                    extracted_url = self._extract_image_url_from_text(part["text"])
                    if extracted_url:
                        image_url = extracted_url
                        logger.info(f"Found image URL in text part (edit): {image_url}")
                        break
            
            # 修改：修改逻辑：如果未找到inlineData或可识别的图片URL，记录警告
            if not image_data and not image_url:
                logger.warning(f"Gemini API edit response missing 'inlineData' or recognizable image URL: {data}")
        else:
            # 修改：修改逻辑：如果未找到candidates或content parts，记录警告
            logger.warning(f"Gemini API edit response missing 'candidates' or content parts for image data: {data}")

        if image_data:
            return image_data
        elif image_url:
            # 如果找到URL，下载图片并返回其字节数据
            temp_download_path = os.path.join(self.save_dir, f"downloaded_edit_{uuid.uuid4()}.png")
            try:
                await self._download_image_from_url(image_url, temp_download_path)
                with open(temp_download_path, "rb") as f:
                    downloaded_image_bytes = f.read()
                logger.info(f"Successfully downloaded image from URL (edit): {image_url}")
                return downloaded_image_bytes
            except Exception as download_e:
                logger.error(f"Failed to download image from URL {image_url} (edit): {download_e}")
                raise Exception(f"编辑图片成功，但从URL下载图片失败: {download_e}")
            finally:
                if os.path.exists(temp_download_path):
                    os.remove(temp_download_path)
                    logger.info(f"Cleaned up temporary downloaded file (edit): {temp_download_path}")
        else:
            raise Exception("编辑图片成功，但未获取到图片数据或图片URL")

    async def _generate_image_manually(self, prompt, api_key):
        """使用异步请求生成图片（替换同步请求）"""
        model_name = "gemini-2.0-flash-preview-image-generation"

        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.info(f"异步请求地址：{endpoint}")

        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024,
            },
        }

        # 异步发送请求
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=endpoint, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()  # 异步读取响应文本
                        logger.error(
                            f"API生成请求失败: HTTP {response.status}, 响应: {response_text}"
                        )
                        response.raise_for_status()

                    # 异步解析JSON响应
                    data = await response.json()
                    # 修改：新增：记录完整的API响应数据，以便调试
                    logger.debug(f"Gemini API generate response data: {data}")

            except Exception as e:
                logger.error(f"异步生成请求失败: {str(e)}")
                raise  # 传递异常触发重试

        # 解析图片数据或URL
        image_data = None
        image_url = None

        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
                    image_data = base64.b64decode(base64_str)
                    logger.info("Successfully extracted image data from inlineData (generate).")
                    break
                # 修改：修改逻辑：调用辅助函数识别多种格式的图片URL
                elif "text" in part:
                    extracted_url = self._extract_image_url_from_text(part["text"])
                    if extracted_url:
                        image_url = extracted_url
                        logger.info(f"Found image URL in text part (generate): {image_url}")
                        break
            
            # 修改：修改逻辑：如果未找到inlineData或可识别的图片URL，记录警告
            if not image_data and not image_url:
                logger.warning(f"Gemini API generate response missing 'inlineData' or recognizable image URL: {data}")
        else:
            # 修改：修改逻辑：如果未找到candidates或content parts，记录警告
            logger.warning(f"Gemini API generate response missing 'candidates' or content parts for image data: {data}")

        if image_data:
            return image_data
        elif image_url:
            # 如果找到URL，下载图片并返回其字节数据
            temp_download_path = os.path.join(self.save_dir, f"downloaded_gen_{uuid.uuid4()}.png")
            try:
                await self._download_image_from_url(image_url, temp_download_path)
                with open(temp_download_path, "rb") as f:
                    downloaded_image_bytes = f.read()
                logger.info(f"Successfully downloaded image from URL (generate): {image_url}")
                return downloaded_image_bytes
            except Exception as download_e:
                logger.error(f"Failed to download image from URL {image_url} (generate): {download_e}")
                raise Exception(f"生成图片成功，但从URL下载图片失败: {download_e}")
            finally:
                if os.path.exists(temp_download_path):
                    os.remove(temp_download_path)
                    logger.info(f"Cleaned up temporary downloaded file (generate): {temp_download_path}")
        else:
            raise Exception("生成图片成功，但未获取到图片数据或图片URL")

    async def terminate(self):
        """插件卸载时清理临时目录"""
        if os.path.exists(self.save_dir):
            try:
                for file in os.listdir(self.save_dir):
                    os.remove(os.path.join(self.save_dir, file))
                os.rmdir(self.save_dir)
                logger.info(f"插件卸载：已清理临时目录 {self.save_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {str(e)}")
        logger.info("Gemini 文生图插件已停用")

